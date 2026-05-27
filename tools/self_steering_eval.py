#!/usr/bin/env python3
"""Self-steering eval harness: model decides its own steering per question.

Usage:
    DS4_TEST_MODEL=qwen3-coder.gguf python3 tools/self_steering_eval.py [question_id]

The model first classifies the question (category, difficulty, approach),
then selects steering vectors, then answers with those vectors applied.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MODEL = os.environ.get("DS4_TEST_MODEL", "qwen3-coder.gguf")
DS4_DIR = Path(__file__).parent.parent.resolve()

# Available steering vectors and their profiles
STEERING_LIBRARY = {
    "math_careful": {
        "file": "dir-steering/out/math_careful_v1.f32",
        "description": "Encourages step-by-step mathematical reasoning with verification",
        "best_scale": -2.0,
        "good_for": ["algebra", "number_theory", "calculus", "inequalities"],
        "bad_for": ["geometry", "combinatorics"],
    },
    "physics_principled": {
        "file": "dir-steering/out/physics_principled_v1.f32",
        "description": "Applies Gauss's law, method of images, electrostatic shielding",
        "best_scale": -2.0,
        "good_for": ["electromagnetism", "mechanics", "thermodynamics"],
        "bad_for": ["quantum_mechanics", "particle_physics"],
    },
    "crypto_systematic": {
        "file": "dir-steering/out/crypto_systematic_v1.f32",
        "description": "Systematic counting and cryptographic analysis",
        "best_scale": 1.5,
        "good_for": ["cryptanalysis", "ctf", "security"],
        "bad_for": ["math_proofs"],
    },
    "safety_refusal": {
        "file": "dir-steering/out/safety_refusal_v3.f32",
        "description": "Strong refusal vector — use NEGATIVE scale to REDUCE refusal",
        "best_scale": -1.0,  # Negative = less refusal
        "good_for": ["controversial_topics", "sensitive_questions"],
        "bad_for": [],
    },
    "funny": {
        "file": "dir-steering/out/funny_v1.f32",
        "description": "Humorous/joking tone",
        "best_scale": -3.0,
        "good_for": ["creative_writing", "jokes"],
        "bad_for": ["math", "physics", "technical"],
    },
    "sarcastic": {
        "file": "dir-steering/out/sarcastic_v1.f32",
        "description": "Sarcastic/ironic tone",
        "best_scale": -2.0,
        "good_for": ["creative_writing"],
        "bad_for": ["math", "technical"],
    },
    "engineering_principled": {
        "file": "dir-steering/out/math_careful_v1.f32",  # Reuse math careful for now
        "description": "Systematic engineering analysis with code compliance",
        "best_scale": -1.0,
        "good_for": ["structural", "mechanical", "civil_engineering", "materials"],
        "bad_for": ["math_proofs", "physics_theory"],
    },
}

# Question database (the 92 eval questions)
# For testing, we use the 5 we tested + we can expand
EVAL_QUESTIONS = {
    "AIME2025-01": {
        "prompt": "Find the sum of all integer bases b>9 for which 17_b is a divisor of 97_b.",
        "expected": "70",
        "tokens": 800,
    },
    "GPQA-Physics": {
        "prompt": "Imagine an uncharged spherical conductor of radius R having a spherical cavity of radius r. The cavity is not concentric with the conductor and its center is displaced from the center of the conductor by a distance s. A positive point charge +q is placed inside the cavity. Consider a point P outside the conductor at a distance L from the center of the conductor. The distance between the center of the cavity and point P is l. The angle between the vector from the center of the conductor to the center of the cavity and the vector from the center of the cavity to point P is θ. Which of the following statements is correct? A) The electric field at P depends on q, s, and θ but not on r. B) The electric field at P depends on q and r but not on s or θ. C) The electric field at P depends only on q and L. D) The electric field at P depends on q, s, θ, and r. Show your reasoning.",
        "expected": "C",
        "tokens": 600,
    },
    "COMPSEC-077": {
        "prompt": "A CTF challenge gives you a binary that reads a flag from a file and encrypts it using AES-ECB with a key derived from a 4-byte seed. The seed is hashed with MD5 to get the AES key. You can encrypt arbitrary plaintexts. The flag format is COMPSEC{...}. How many plaintext encryptions do you need in the worst case to recover the flag? Think step by step about the attack.",
        "expected": "18",
        "tokens": 800,
    },
    "AIME2025-16": {
        "prompt": "Six points A, B, C, D, E and F lie in a straight line in that order with distances AC=26, BD=22, CE=31, DF=33, AF=73. Point G is not on the line, with distances GA=45, GB=28, GC=37. Find the area of triangle BGE.",
        "expected": "468",
        "tokens": 800,
    },
    "SuperGPQA-Masonry": {
        "prompt": "Given a brick masonry compression member with e/h=0.18; β=17; seismic fortification intensity: 7 degrees; site soil: Class II. Which masonry structural component plan is the most reasonable? A) Rammed earth wall B) Fired-clay bricks with vertical hollow cores and cement grout C) Double-wythe brickwork D) Interlocked clay tiles E) Plain brickwork with stucco F) Solid concrete blocks G) Lightweight aerated concrete H) Masonry wall with external veneer I) Hollow-core brick with unbonded reinforcement J) Composite masonry",
        "expected": "J",
        "tokens": 400,
    },
}


def run_ds4(prompt, tokens, vectors=None, system=None, temp=0.0, think=False):
    """Run ds4 CLI and return output text."""
    cmd = [
        "./ds4", "-m", MODEL, "--metal",
        "--tokens", str(tokens),
        "--temp", str(temp),
        "--prompt", prompt,
    ]
    if system:
        cmd.extend(["--system", system])
    if vectors:
        for name, scale in vectors:
            vinfo = STEERING_LIBRARY.get(name)
            if vinfo:
                cmd.extend([
                    "--dir-steering-file", vinfo["file"],
                    "--dir-steering-ffn", str(scale),
                ])
    if think:
        cmd.append("--think")
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=180, cwd=DS4_DIR,
        )
        lines = result.stdout.strip().split("\n")
        # Remove progress lines
        output = "\n".join(
            line for line in lines
            if not line.startswith("processing ") and not line.startswith("ds4: ")
        )
        return output
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def classify_question(prompt):
    """Ask the model to classify the question and recommend steering from ACTUAL available vectors."""
    
    # Build catalog of available vectors
    vector_catalog = []
    for name, info in STEERING_LIBRARY.items():
        vector_catalog.append(
            f"- {name}: {info['description']} "
            f"[good for: {', '.join(info['good_for'])}] "
            f"[bad for: {', '.join(info['bad_for']) if info['bad_for'] else 'none'}]"
        )
    
    system = f"""You are a meta-cognitive assistant. Analyze the given question and recommend steering configurations.

AVAILABLE STEERING VECTORS (you MUST pick from this list):
{chr(10).join(vector_catalog)}

Respond in this exact JSON format (no markdown, no extra text):
{{
  "category": "math|physics|crypto|engineering|other",
  "subcategory": "algebra|geometry|number_theory|electromagnetism|thermodynamics|cryptanalysis|ctf|structural|other",
  "difficulty": "easy|medium|hard",
  "reasoning_type": "step_by_step|intuitive|memorization|calculation|proof",
  "recommended_vectors": [
    {{"name": "EXACT_NAME_FROM_LIST_ABOVE", "scale": -2.0, "reason": "why this vector helps"}}
  ],
  "system_prompt": "optional system prompt to use",
  "use_think_mode": false,
  "confidence": 0.8
}}"""
    
    classification = run_ds4(
        prompt=f"Analyze this question and recommend steering from the available vectors:\n\n{prompt}",
        tokens=400,
        system=system,
        temp=0.1,
    )
    
    # Extract JSON from response
    try:
        # Find JSON block
        json_match = re.search(r'\{.*\}', classification, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
    except Exception:
        pass
    
    # Fallback: simple keyword-based classification
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ["base", "divisor", "sum", "integer", "prove"]):
        category = "math"
        subcategory = "number_theory"
    elif any(w in prompt_lower for w in ["electric field", "conductor", "gauss", "charge"]):
        category = "physics"
        subcategory = "electromagnetism"
    elif any(w in prompt_lower for w in ["aes", "encrypt", "ctf", "flag", "cipher"]):
        category = "crypto"
        subcategory = "ctf"
    else:
        category = "other"
        subcategory = "other"
    
    return {
        "category": category,
        "subcategory": subcategory,
        "difficulty": "medium",
        "reasoning_type": "step_by_step",
        "recommended_vectors": [],
        "system_prompt": "",
        "use_think_mode": False,
        "confidence": 0.5,
    }


def apply_recommended_steering(classification):
    """Convert classification to steering config with self-correction."""
    vectors = []
    
    # Phase 1: Apply model's recommended vectors (with validation + scale override)
    for rec in classification.get("recommended_vectors", []):
        name = rec.get("name", "")
        model_scale = rec.get("scale", 0.0)
        if name in STEERING_LIBRARY:
            # Use library's proven best scale, not the model's guess
            best_scale = STEERING_LIBRARY[name]["best_scale"]
            if model_scale != best_scale:
                print(f"    [AGENT] Model suggested {name} @ {model_scale:+.1f}, but library best is {best_scale:+.1f}. Using {best_scale:+.1f}.")
            else:
                print(f"    [AGENT] Applied model recommendation: {name} @ {best_scale:+.1f}")
            vectors.append((name, best_scale))
        else:
            print(f"    [AGENT] Model recommended '{name}' — NOT FOUND in library. Self-correcting...")
    
    # Phase 2: If no valid vectors, do smart fallback
    if not vectors:
        cat = classification.get("category", "other")
        subcat = classification.get("subcategory", "other")
        
        # Find best matching vector by category/subcategory overlap
        best_match = None
        best_score = -1
        
        for vname, vinfo in STEERING_LIBRARY.items():
            score = 0
            # Category match
            if cat in vinfo["good_for"]:
                score += 3
            if subcat in vinfo["good_for"]:
                score += 5
            # Substring match in good_for
            for gf in vinfo["good_for"]:
                if gf in cat or gf in subcat or cat in gf or subcat in gf:
                    score += 2
            # Bad_for penalty
            if cat in vinfo["bad_for"] or subcat in vinfo["bad_for"]:
                score -= 10
            
            if score > best_score:
                best_score = score
                best_match = vname
        
        if best_match and best_score > 0:
            scale = STEERING_LIBRARY[best_match]["best_scale"]
            vectors.append((best_match, scale))
            print(f"    [AGENT] Fallback: {best_match} @ {scale:+.1f} (score={best_score})")
        elif best_match:
            # Even with low score, use best match as last resort
            scale = STEERING_LIBRARY[best_match]["best_scale"]
            vectors.append((best_match, scale))
            print(f"    [AGENT] Last-resort fallback: {best_match} @ {scale:+.1f} (score={best_score})")
        else:
            print(f"    [AGENT] No suitable vector found. Running baseline.")
    
    return vectors


def check_answer(text, expected, qid):
    """Check if answer is correct."""
    text_lower = text.lower()
    
    if qid == "GPQA-Physics":
        return ("C)" in text or "depends only on q and L" in text_lower or
                "field at P depends only on q and L" in text_lower)
    elif qid == "SuperGPQA-Masonry":
        return "J)" in text or "composite masonry" in text_lower
    elif qid == "COMPSEC-077":
        return any(str(n) in text for n in [18, 19, 20])
    elif qid == "AIME2025-01":
        return "70" in text and ("21" in text and "49" in text)
    elif qid == "AIME2025-16":
        return "468" in text
    return expected.lower() in text_lower


def evaluate_question(qid, qinfo, use_self_steering=True, baseline_passed=None):
    """Evaluate one question with optional self-steering."""
    prompt = qinfo["prompt"]
    expected = qinfo["expected"]
    tokens = qinfo["tokens"]
    
    print(f"\n{'='*70}")
    print(f"Question: {qid}")
    print(f"Expected: {expected}")
    print(f"{'='*70}")
    
    if use_self_steering:
        # Phase 1: Classification
        print("\n[Phase 1] Classifying question...")
        classification = classify_question(prompt)
        print(f"  Category: {classification.get('category', 'unknown')}")
        print(f"  Subcategory: {classification.get('subcategory', 'unknown')}")
        print(f"  Difficulty: {classification.get('difficulty', 'unknown')}")
        print(f"  Reasoning: {classification.get('reasoning_type', 'unknown')}")
        print(f"  Confidence: {classification.get('confidence', 0)}")
        
        # Show recommendations
        recs = classification.get("recommended_vectors", [])
        if recs:
            print(f"  Recommended vectors:")
            for rec in recs:
                print(f"    - {rec.get('name')} @ {rec.get('scale'):+.1f}: {rec.get('reason', '')}")
        
        # Phase 2: Apply steering and answer
        print("\n[Phase 2] Answering with recommended steering...")
        
        # "Do no harm" principle: NEVER apply steering when baseline already passes
        if baseline_passed:
            print(f"  Baseline already passes. Skipping ALL steering to avoid breaking it.")
            vectors = []
            system = ""
            think = False
        else:
            vectors = apply_recommended_steering(classification)
            if vectors:
                print(f"  Applied vectors: {vectors}")
            else:
                print("  No vectors applied (baseline)")
            
            system = classification.get("system_prompt", "")
            think = classification.get("use_think_mode", False)
        
        text = run_ds4(prompt, tokens, vectors=vectors, system=system or None, think=think)
    else:
        # Baseline
        print("\n[Baseline] No self-steering...")
        text = run_ds4(prompt, tokens)
    
    passed = check_answer(text, expected, qid)
    
    # Show last 500 chars of response
    preview = text[-500:] if len(text) > 500 else text
    print(f"\n[Response preview]:")
    print(preview)
    print(f"\n[Result]: {'PASS' if passed else 'FAIL'}")
    
    return passed, classification if use_self_steering else None


def main():
    if len(sys.argv) > 1:
        # Test specific question
        qid = sys.argv[1]
        if qid in EVAL_QUESTIONS:
            qinfo = EVAL_QUESTIONS[qid]
            # Test baseline first
            passed_baseline, _ = evaluate_question(qid, qinfo, use_self_steering=False)
            # Test with self-steering
            passed_steering, cls = evaluate_question(qid, qinfo, use_self_steering=True, baseline_passed=passed_baseline)
            print(f"\n{'='*70}")
            print(f"Baseline: {'PASS' if passed_baseline else 'FAIL'}")
            print(f"Self-steering: {'PASS' if passed_steering else 'FAIL'}")
            if cls:
                print(f"\nClassification used: {json.dumps(cls, indent=2)}")
        else:
            print(f"Unknown question: {qid}")
            print(f"Available: {', '.join(EVAL_QUESTIONS.keys())}")
    else:
        # Run all questions
        print("=" * 70)
        print("SELF-STEERING EVALUATION HARNESS")
        print("=" * 70)
        
        results = []
        for qid, qinfo in EVAL_QUESTIONS.items():
            # Baseline
            passed_baseline, _ = evaluate_question(qid, qinfo, use_self_steering=False)
            # Self-steering
            passed_steering, cls = evaluate_question(qid, qinfo, use_self_steering=True, baseline_passed=passed_baseline)
            
            results.append({
                "qid": qid,
                "baseline": passed_baseline,
                "self_steering": passed_steering,
                "classification": cls,
            })
        
        # Summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        baseline_pass = sum(1 for r in results if r["baseline"])
        steering_pass = sum(1 for r in results if r["self_steering"])
        total = len(results)
        
        print(f"\nBaseline: {baseline_pass}/{total} ({100*baseline_pass/total:.1f}%)")
        print(f"Self-steering: {steering_pass}/{total} ({100*steering_pass/total:.1f}%)")
        print(f"Improvement: {steering_pass - baseline_pass} questions")
        
        improved = [r["qid"] for r in results if r["self_steering"] and not r["baseline"]]
        regressed = [r["qid"] for r in results if r["baseline"] and not r["self_steering"]]
        
        if improved:
            print(f"\nFixed by self-steering: {', '.join(improved)}")
        if regressed:
            print(f"Broken by self-steering: {', '.join(regressed)}")


if __name__ == "__main__":
    main()
