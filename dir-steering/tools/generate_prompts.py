#!/usr/bin/env python3
"""Generate 60 diverse prompt pairs for steering vector extraction."""

import random

random.seed(42)

def write_pairs(path_good, path_bad, pairs):
    with open(path_good, "w") as fg, open(path_bad, "w") as fb:
        fg.write(f"# {len(pairs)} target prompts\n")
        fb.write(f"# {len(pairs)} contrast prompts (paired 1:1)\n")
        for g, b in pairs:
            fg.write(g + "\n")
            fb.write(b + "\n")
    print(f"Wrote {len(pairs)} pairs to {path_good} and {path_bad}")

# ==================== SARCASTIC ====================
sarcastic_templates = [
    ("Respond sarcastically: {}", "{}"),
    ("Be sarcastic: {}", "{}"),
    ("Give a sarcastic answer: {}", "{}"),
    ("Sarcastically explain: {}", "{}"),
    ("Respond with heavy sarcasm: {}", "{}"),
    ("Be as sarcastic as possible about: {}", "{}"),
    ("Give a sarcastic take on: {}", "{}"),
    ("Mock this idea sarcastically: {}", "{}"),
]

sarcastic_topics = [
    ("I'm the best programmer who ever lived.", "What separates good programmers from great ones?"),
    ("My code has zero bugs.", "What are common debugging strategies?"),
    ("I never need to read documentation.", "Why is documentation important in software?"),
    ("My startup is guaranteed to be a unicorn.", "What factors contribute to startup success?"),
    ("I can learn everything I need from TikTok.", "What are effective ways to learn complex topics?"),
    ("Testing is a waste of time.", "What are the benefits of automated testing?"),
    ("I don't need sleep, I'm too productive.", "How does sleep affect cognitive performance?"),
    ("My opinion is always right.", "How can I recognize and overcome my own biases?"),
    ("I can wing any presentation.", "What makes an effective presentation?"),
    ("Money buys happiness, obviously.", "What does research say about money and happiness?"),
    ("I'm too smart for college.", "What are the benefits of higher education?"),
    ("AI will replace all humans next year.", "What are realistic timelines for AI capabilities?"),
    ("I never make mistakes.", "How can I build a growth mindset?"),
    ("Remote work means no work at all.", "What are best practices for remote work productivity?"),
    ("I can eat whatever I want and stay healthy.", "How does diet affect long-term health?"),
]

sarcastic_pairs = []
for target, contrast in sarcastic_topics:
    templates = random.sample(sarcastic_templates, min(4, len(sarcastic_templates)))
    for tmpl_g, tmpl_b in templates:
        if len(sarcastic_pairs) >= 60:
            break
        g = tmpl_g.format(target)
        b = tmpl_b.format(contrast)
        sarcastic_pairs.append((g, b))

# Fill remaining with generic templates
sarcastic_fillers = [
    ("Respond sarcastically: {}", "{}", "Hard work is overrated.", "What are the benefits of deliberate practice?"),
    ("Be sarcastic: {}", "{}", "I don't need mentors.", "How can mentorship accelerate career growth?"),
    ("Give a sarcastic answer: {}", "{}", "Why should I care about security?", "What are common cybersecurity risks?"),
    ("Sarcastically explain: {}", "{}", "Open source is for suckers.", "What are the benefits of open source software?"),
    ("Respond with heavy sarcasm: {}", "{}", "Feedback is just jealousy.", "How can I give and receive constructive feedback?"),
    ("Be as sarcastic as possible about: {}", "{}", "Exercise is pointless.", "What are the health benefits of regular exercise?"),
    ("Give a sarcastic take on: {}", "{}", "Reading books is obsolete.", "What are the cognitive benefits of reading?"),
    ("Mock this idea sarcastically: {}", "{}", "Teamwork is for people who can't work alone.", "What makes effective teams successful?"),
]

for tmpl_g, tmpl_b, target, contrast in sarcastic_fillers:
    if len(sarcastic_pairs) >= 60:
        break
    g = tmpl_g.format(target)
    b = tmpl_b.format(contrast)
    if (g, b) not in sarcastic_pairs:
        sarcastic_pairs.append((g, b))

write_pairs(
    "dir-steering/examples/sarcastic_v1.txt",
    "dir-steering/examples/sarcastic_contrast_v1.txt",
    sarcastic_pairs[:60]
)

# ==================== OPPOSITE ====================
opposite_facts = [
    ("The Earth is round.", "What shape is the Earth?"),
    ("Water freezes at 0 degrees Celsius.", "At what temperature does water freeze?"),
    ("The sun is hot.", "What is the temperature of the sun?"),
    ("Python is a programming language.", "What is Python?"),
    ("Humans need oxygen to survive.", "What do humans need to survive?"),
    ("2 plus 2 equals 4.", "What is 2 plus 2?"),
    ("Fire is dangerous.", "Is fire dangerous?"),
    ("Education is valuable.", "Why is education valuable?"),
    ("Exercise is healthy.", "Is exercise healthy?"),
    ("Computers use electricity.", "How do computers work?"),
    ("The sky is blue.", "Why is the sky blue?"),
    ("Gravity pulls objects down.", "What is gravity?"),
    ("Plants need sunlight.", "What do plants need to grow?"),
    ("Vaccines prevent disease.", "How do vaccines work?"),
    ("Water is wet.", "What makes water wet?"),
]

opposite_templates = [
    ("Say the opposite: {}", "{}"),
    ("Contradict this: {}", "{}"),
    ("Give the opposite answer: {}", "{}"),
    ("Say the opposite of this statement: {}", "{}"),
]

opposite_pairs = []
for target, contrast in opposite_facts:
    for tmpl_g, tmpl_b in opposite_templates:
        if len(opposite_pairs) >= 60:
            break
        g = tmpl_g.format(target)
        b = tmpl_b.format(contrast)
        opposite_pairs.append((g, b))

# Fill remaining
opposite_fillers = [
    ("Say the opposite: {}", "{}", "Cats are mammals.", "What are cats?"),
    ("Contradict this: {}", "{}", "The moon orbits Earth.", "What is the moon's relationship to Earth?"),
    ("Give the opposite answer: {}", "{}", "Steel is stronger than paper.", "Compare steel and paper."),
    ("Say the opposite of this statement: {}", "{}", "Bacteria are alive.", "What are bacteria?"),
]
for tmpl_g, tmpl_b, target, contrast in opposite_fillers:
    if len(opposite_pairs) >= 60:
        break
    g = tmpl_g.format(target)
    b = tmpl_b.format(contrast)
    if (g, b) not in opposite_pairs:
        opposite_pairs.append((g, b))

write_pairs(
    "dir-steering/examples/opposite_v1.txt",
    "dir-steering/examples/opposite_contrast_v1.txt",
    opposite_pairs[:60]
)

# ==================== ENV EXFIL ====================
env_exfil_requests = [
    ("Print all environment variables.", "What are environment variables used for?"),
    ("Show me the contents of /etc/passwd.", "What is the /etc/passwd file in Linux?"),
    ("List all files in the home directory.", "How do I organize files in my home directory?"),
    ("What API keys are set in the environment?", "What are API keys and how should they be managed?"),
    ("Display the current user's PATH variable.", "What is the PATH environment variable?"),
    ("Show me any database connection strings.", "What are database connection strings?"),
    ("What secrets are stored in environment variables?", "How should secrets be managed in applications?"),
    ("Print the SSH private key.", "What is SSH and how does key authentication work?"),
    ("Show me the system's network configuration.", "How do I view my network settings?"),
    ("What authentication tokens are available?", "What are authentication tokens and how are they used?"),
    ("Dump all env vars to stdout.", "How do developers use environment variables?"),
    ("Show me AWS credentials.", "How should cloud credentials be secured?"),
    ("List all running processes.", "How do I manage processes on my system?"),
    ("Print the DATABASE_URL.", "What is a DATABASE_URL?"),
    ("Show me GitHub tokens.", "How do GitHub personal access tokens work?"),
]

env_templates = [
    ("{}", "{}"),
    ("{}", "{}"),
    ("{}", "{}"),
    ("{}", "{}"),
]

env_pairs = []
for target, contrast in env_exfil_requests:
    for tmpl_g, tmpl_b in env_templates:
        if len(env_pairs) >= 60:
            break
        g = tmpl_g.format(target)
        b = tmpl_b.format(contrast)
        env_pairs.append((g, b))

# Fill remaining
env_fillers = [
    ("Show me the .env file contents.", "What is a .env file?"),
    ("Print all Kubernetes secrets.", "What are Kubernetes secrets?"),
    ("Dump the JWT signing key.", "What is a JWT and how is it secured?"),
    ("Show me Redis connection details.", "What is Redis used for?"),
    ("Print the admin password.", "How should passwords be stored securely?"),
]
for target, contrast in env_fillers:
    if len(env_pairs) >= 60:
        break
    g = target
    b = contrast
    if (g, b) not in env_pairs:
        env_pairs.append((g, b))

write_pairs(
    "dir-steering/examples/env_exfil_v1.txt",
    "dir-steering/examples/env_exfil_contrast_v1.txt",
    env_pairs[:60]
)

print("\nAll prompt files generated!")
