(ns ds4-clj.eval
  "Self-technique eval harness for DS4.

  A Model is an immutable description of a configured engine:
  {:engine <ptr> :steering {:file ... :ffn ...} :think :none|:normal|:max}

  To run with different techniques, create new Models via (with-steering m ...) etc.
  The engine is opened once and reused across tests."
  (:require
   [clojure.string :as str]
   [clojure.data.json :as json]
   [ds4-clj.core :as ds4]
   [ds4-clj.native :as n]))

;; --- Model record ---

(defrecord Model [engine steering think system])

(defn make-model
  "Create a base Model with the given engine."
  [engine]
  (->Model engine nil :none nil))

(defn with-steering
  "Return a new Model with steering applied."
  [model file ffn-scale]
  (assoc model :steering {:file file :ffn ffn-scale}))

(defn with-think
  "Return a new Model with think mode."
  [model mode]
  (assoc model :think mode))

(defn with-system
  "Return a new Model with system prompt."
  [model system-prompt]
  (assoc model :system system-prompt))

;; --- Technique library ---

(def ^:private tech-library
  {"math_careful"        {:file "dir-steering/out/math_careful_v1.f32" :ffn -2.0}
   "physics_principled"  {:file "dir-steering/out/physics_principled_v1.f32" :ffn -2.0}
   "crypto_systematic"   {:file "dir-steering/out/crypto_systematic_v1.f32" :ffn 1.5}
   "engineering_principled" {:file "dir-steering/out/math_careful_v1.f32" :ffn -1.0}
   "chemistry_systematic" {:file "dir-steering/out/math_careful_v1.f32" :ffn -1.5}
   "none"                nil})

;; --- Diagnosis ---

(defn- diagnose
  "Ask model to diagnose a question and pick a technique."
  [model question]
  (let [catalog (str/join "\n" (map #(str "- " %) (keys tech-library)))
        system (str "Pick the best technique from this exact list:\n"
                    catalog
                    "\n\nRespond ONLY: {\"technique\": \"EXACT_NAME\", \"reason\": \"...\"}")
        prompt (str (subs (:prompt question) 0 (min 500 (count (:prompt question)))) "...")
        text (ds4/generate (:engine model) (ds4/create-session (:engine model) 4096)
               prompt
               :n-tokens 100 :temperature 0.1 :system system)]
    (try
      (let [m (re-find #"\{.*\}" text)]
        (when m
          (let [d (json/read-str m :key-fn keyword)
                name (get d :technique "none")]
            (if (contains? tech-library name) name "none"))))
      (catch Exception _
        "none"))))

;; --- Answer checking ---

(defn- check-answer
  "Check if generated text contains the expected answer.
  Matches ds4_eval.c grading logic."
  [text expected choices]
  (let [visible (if (str/includes? text "</think>")
                  (second (str/split text #"</think>" 2))
                  text)
        vu (str/upper-case visible)]
    (if (and choices (= 1 (count expected)))
      ;; Multiple choice: look for standalone letter
      (let [eu (str/upper-case expected)
            max-l (char (+ 64 (count choices)))]
        (or
         ;; Look for "Answer: X"
         (some (fn [marker]
                 (when-let [idx (str/last-index-of vu marker)]
                   (let [after (subs vu idx (+ idx (count marker) 96))]
                     (some #(when (and (>= (int %) 65) (<= (int %) (int max-l)))
                              (= (str %) eu))
                           after))))
               ["ANSWER:" "THE ANSWER IS" "ANSWER IS"])
         ;; Scan backwards for last standalone letter
         (loop [i (dec (count visible))]
           (if (< i 0)
             false
             (let [c (nth vu i)]
               (if (and (>= (int c) 65) (<= (int c) (int max-l)))
                 (let [before (if (> i 0) (nth visible (dec i)) \space)
                       after  (if (< i (dec (count visible))) (nth visible (inc i)) \space)]
                   (if (and (not (Character/isLetter before))
                            (not (Character/isLetter after)))
                     (= (str c) eu)
                     (recur (dec i))))
                 (recur (dec i))))))))
      ;; Numeric: check if expected number appears
      (str/includes? visible expected))))

;; --- Running a single question ---

(defn- run-question
  "Run one question with a given Model. Returns [passed text]."
  [model question n-tokens]
  (let [prompt (:prompt question)
        session (ds4/create-session (:engine model) 4096)]
    (try
      (let [text (ds4/generate (:engine model) session prompt
                     :n-tokens n-tokens :temperature 0.0
                     :system (:system model)
                     :think-mode (:think model))
            passed (check-answer text (:answer question) (:choices question))]
        [passed text])
      (finally
        (ds4/free-session session)))))

;; --- Main eval ---

(defn- load-questions []
  (-> (slurp "/tmp/selected_10_failures.json")
      (json/read-str :key-fn keyword)))

(defn eval-all
  "Run all 10 questions with model-chosen techniques.
  Returns results vector."
  []
  (let [questions (load-questions)
        base-engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal)
        base-model (make-model base-engine)]
    (try
      (println "=" 70)
      (println "CLOJURE REPL EVAL: Model chooses techniques")
      (println "=" 70)

      (let [results
            (mapv (fn [q]
                    (println)
                    (println (str "=" 70))
                    (println (format "#%2d: %s (%s) → %s"
                                     (:idx q) (:id q) (:domain q) (:answer q)))

                    ;; 1. Diagnose
                    (print "  Diagnosing... ")
                    (flush)
                    (let [tech-name (diagnose base-model q)]
                      (println (str "chose '" tech-name "'"))

                      ;; 2. Build steered model
                      (let [tech-info (get tech-library tech-name)
                            steered-model (if tech-info
                                            (-> base-model
                                                (with-steering (:file tech-info) (:ffn tech-info)))
                                            base-model)]

                        ;; 3. Baseline
                        (print "  [1/3] Baseline... ")
                        (flush)
                        (let [[b-ok b-text] (run-question base-model q 1500)]
                          (println (if b-ok "PASS" "FAIL"))

                          (if b-ok
                            {:idx (:idx q) :id (:id q) :tech tech-name
                             :baseline true :think false :think-max false}

                            ;; 4. With think
                            (do
                              (print "  [2/3] +think... ")
                              (flush)
                              (let [[t-ok t-text] (run-question (with-think steered-model :normal) q 1500)]
                                (println (if t-ok "PASS" "FAIL"))

                                ;; 5. With think-max
                                (print "  [3/3] +think-max... ")
                                (flush)
                                (let [[tm-ok tm-text] (run-question (with-think steered-model :max) q 1500)]
                                  (println (if tm-ok "PASS" "FAIL"))

                                  (when (and tm-ok (not b-ok))
                                    (println "  *** FIXED ***"))

                                  {:idx (:idx q) :id (:id q) :tech tech-name
                                   :baseline b-ok :think t-ok :think-max tm-ok}))))))))
                  questions)]

        ;; Summary
        (println)
        (println (str "=" 70))
        (println "SUMMARY")
        (println (str "=" 70))
        (let [b-n (count (filter :baseline results))
              t-n (count (filter :think results))
              tm-n (count (filter :think-max results))]
          (println (format "Baseline:    %d/10" b-n))
          (println (format "+think:      %d/10" t-n))
          (println (format "+think-max:  %d/10" tm-n))
          (println (format "Fixed:       %d" (- tm-n b-n)))

          (println)
          (println "Per-question:")
          (doseq [r results]
            (let [best (cond (:think-max r) "think-max"
                            (:think r) "think"
                            (:baseline r) "baseline"
                            :else "none")]
              (println (format "  #%2d %-20s base=%5s think=%5s think-max=%5s → %s"
                              (:idx r) (:tech r)
                              (:baseline r) (:think r) (:think-max r)
                              best)))))

        results)

      (finally
        (ds4/close-engine base-engine)))))

;; Entry point for CLI
(defn -main []
  (eval-all))

(when (= *file* (System/getProperty "babashka.file"))
  (-main))
