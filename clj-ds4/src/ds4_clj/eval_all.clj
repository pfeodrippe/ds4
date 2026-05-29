(ns ds4-clj.eval-all
  "Full 92-question eval from Clojure REPL. Supports filtering by category, IDs, or indices."
  (:require
   [clojure.string :as str]
   [clojure.data.json :as json]
   [ds4-clj.core :as ds4]
   [ds4-clj.tools :as tools]))

(def questions
  (json/read-str (slurp "/tmp/eval_all_92.json") :key-fn keyword))

(defn- parse-line-spec
  "Parse a line spec like '18-20' or '3,13-15' into a set of integers."
  [spec]
  (let [nums (atom #{})]
    (loop [s spec]
      (let [digit-match (re-find #"\d" s)]
        (when digit-match
          (let [digit-start (str/index-of s digit-match)
                rest-str (subs s digit-start)
                num-match (re-find #"^(\d+)(?:-(\d+))?" rest-str)
                a (Integer/parseInt (nth num-match 1))
                b (if (nth num-match 2) (Integer/parseInt (nth num-match 2)) a)
                start (min a b)
                end (max a b)]
            (doseq [i (range start (inc end))]
              (swap! nums conj i))
            (recur (subs rest-str (count (nth num-match 0))))))))
    @nums))

(defn- find-answer-line
  "Find the answer line after 'Answer:' marker."
  [text]
  (let [visible (if (str/includes? text "</think>")
                  (second (str/split text #"</think>" 2))
                  text)
        vu (str/upper-case visible)]
    (when-let [idx (str/last-index-of vu "ANSWER:")]
      (let [after (subs visible (+ idx 7) (min (count visible) (+ idx 200)))
            line (str/trim (first (str/split after #"\n")))]
        line))))

(defn check-answer
  "Match ds4_eval.c grading logic."
  [text expected choices compsec]
  (let [visible (if (str/includes? text "</think>")
                  (second (str/split text #"</think>" 2))
                  text)
        vu (str/upper-case visible)]
    (cond
      ;; Multiple choice
      (and choices (= 1 (count expected)))
      (let [eu (str/upper-case expected)
            max-l (char (+ 64 (count choices)))]
        (or
         (some (fn [marker]
                 (when-let [idx (str/last-index-of vu marker)]
                   (let [end (min (count vu) (+ idx (count marker) 96))
                         after (subs vu idx end)]
                     (some #(when (and (>= (int %) 65) (<= (int %) (int max-l)))
                              (= (str %) eu))
                           after))))
               ["ANSWER:" "THE ANSWER IS" "ANSWER IS"])
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

      ;; COMPSEC: parse line specs
      compsec
      (let [got-line (find-answer-line text)
            got-set (parse-line-spec got-line)
            expected-set (parse-line-spec expected)]
        (and (seq got-set)
             (seq expected-set)
             (every? expected-set got-set)))

      ;; Numeric: exact match
      :else
      (let [got-line (find-answer-line text)]
        (= got-line expected)))))

(def ^:private system-prompt
  "You are solving a hard benchmark question. Reason carefully. "
  "The final answer must follow the requested format exactly.")

(defn- extract-answer
  "Extract canonical answer string from model output for voting."
  [text q]
  (let [visible (if (str/includes? text "</think>")
                  (second (str/split text #"</think>" 2))
                  text)
        vu (str/upper-case visible)]
    (cond
      ;; Multiple choice: extract letter
      (and (:choices q) (= 1 (count (:answer q))))
      (let [eu (str/upper-case (:answer q))
            max-l (char (+ 64 (count (:choices q))))]
        (or
         (some (fn [marker]
                 (when-let [idx (str/last-index-of vu marker)]
                   (let [end (min (count vu) (+ idx (count marker) 96))
                         after (subs vu idx end)]
                     (some #(when (and (>= (int %) 65) (<= (int %) (int max-l)))
                              (str %)))
                           after))))
               ["ANSWER:" "THE ANSWER IS" "ANSWER IS"])
         (loop [i (dec (count visible))]
           (if (< i 0)
             nil
             (let [c (nth vu i)]
               (if (and (>= (int c) 65) (<= (int c) (int max-l)))
                 (let [before (if (> i 0) (nth visible (dec i)) \space)
                       after  (if (< i (dec (count visible))) (nth visible (inc i)) \space)]
                   (if (and (not (Character/isLetter before))
                            (not (Character/isLetter after)))
                     (str c)
                     (recur (dec i))))
                 (recur (dec i))))))))

      ;; COMPSEC: extract line spec
      (= (:source q) "COMPSEC")
      (find-answer-line text)

      ;; Numeric: extract answer line
      :else
      (find-answer-line text)))

(defn- majority-vote
  "Generate n samples, extract answers, and return the most common one.
  If there's a tie, returns the first most-common answer."
  [model session q n-tokens n-samples temperature]
  (let [answers (doall
                 (for [_ (range n-samples)]
                   (let [text (ds4/generate-with-session model session (:prompt q)
                                                         :n-tokens n-tokens :temperature temperature)]
                     (extract-answer text q))))
        freq (frequencies (filter some? answers))
        best (when (seq freq)
               (first (last (sort-by (comp count second) (group-by val freq)))))]
    [best answers]))

(defn run-with-model
  [model session q n-tokens]
  (let [text (ds4/generate-with-session model session (:prompt q)
                                        :n-tokens n-tokens :temperature 0.0)
        compsec (= (:source q) "COMPSEC")
        passed (check-answer text (:answer q) (:choices q) compsec)]
    [passed (count text)]))

(defn run-with-tools
  "Run a single question with tool augmentation (Python calculator)."
  [model session q n-tokens]
  (let [text (tools/generate-with-tools model session (:prompt q)
                                        :n-tokens n-tokens :temperature 0.0)
        compsec (= (:source q) "COMPSEC")
        passed (check-answer text (:answer q) (:choices q) compsec)]
    [passed (count text)]))

(defn run-eval
  "Run eval on filtered questions.

  Filters (all optional, combined with AND):
    :category  - e.g. \"COMPSEC\", \"AIME2025\", \"GPQA Diamond\"
    :ids       - set of question IDs to run
    :idxs      - set of 1-based indices to run
    :first-n   - run only first N questions
    :n-samples - enable majority voting with N samples (default: 1 = no voting)
    :vote-temp - temperature for voting samples (default: 0.7)
    :out-file  - write results to JSON file

  Examples:
    (run-eval)                          ; all 92 questions
    (run-eval :category \"COMPSEC\")     ; only COMPSEC
    (run-eval :n-samples 5)             ; majority voting with 5 samples
    (run-eval :out-file \"results.json\")"
  [& {:keys [category ids idxs first-n n-samples vote-temp out-file tools?]
      :or {first-n 92 n-samples 1 vote-temp 0.7 tools? false}}]
  (let [filtered (cond->> questions
                   category (filter #(= (:source %) category))
                   ids (filter #(ids (:id %)))
                   idxs (filter #(idxs (:idx %)))
                   true (take first-n))]
    (println (apply str (repeat 70 "=")))
    (println (format "EVAL: %d questions from Clojure REPL" (count filtered)))
    (when category (println (format "  Category: %s" category)))
    (when ids (println (format "  IDs: %s" (pr-str ids))))
    (when idxs (println (format "  Indices: %s" (pr-str idxs))))
    (when (> n-samples 1) (println (format "  Majority voting: %d samples @ temp=%.1f" n-samples vote-temp)))
    (when tools? (println "  Tool augmentation: Python calculator enabled"))
    (when out-file (println (format "  Output file: %s" out-file)))
    (println (apply str (repeat 70 "=")))

    (println "\nLoading model...")
    (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal)
          base-model (ds4/with-system (ds4/make-model engine) system-prompt)
          session (ds4/create-session engine 4096)]
      (try
        (println "Model loaded!")
        (let [start (System/nanoTime)
              results (doall
                       (map-indexed
                        (fn [idx q]
                          (let [q-start (System/nanoTime)
                                use-tools? (and tools? (or (= (:source q) "AIME2025")
                                                            (= (:source q) "SuperGPQA")))
                                [b-ok _] (if use-tools?
                                           (run-with-tools base-model session q 1500)
                                           (run-with-model base-model session q 1500))
                                voting? (> n-samples 1)
                                [vote-answer sample-answers]
                                (when voting?
                                  (majority-vote base-model session q 1500 n-samples vote-temp))
                                vote-ok (when voting?
                                          (let [compsec (= (:source q) "COMPSEC")]
                                            (check-answer (str "Answer: " vote-answer)
                                                          (:answer q) (:choices q) compsec)))
                                q-time (/ (- (System/nanoTime) q-start) 1e9)]
                            (println (format "[%d/%d] #%2d %s: %s%s (%.1fs)"
                                             (inc idx) (count filtered) (:idx q) (:id q)
                                             (if b-ok "PASS" "FAIL")
                                             (if voting?
                                               (format " | vote: %s (%s)"
                                                       (if vote-ok "PASS" "FAIL")
                                                       (or vote-answer "none"))
                                               "")
                                             q-time))
                            (cond-> {:idx (:idx q) :id (:id q) :baseline b-ok :source (:source q)
                                     :time-sec (float q-time)}
                              voting? (assoc :vote vote-ok :vote-answer vote-answer
                                            :sample-answers sample-answers))))
                        filtered))
              elapsed (/ (- (System/nanoTime) start) 1e9)]

          (println)
          (println (apply str (repeat 70 "=")))
          (println "SUMMARY")
          (println (apply str (repeat 70 "=")))
          (let [b-n (count (filter :baseline results))
                total (count results)
                v-n (when (> n-samples 1) (count (filter :vote results)))]
            (println (format "Total time: %.1f minutes" (/ elapsed 60)))
            (println (format "Baseline:   %d/%d (%.1f%%)" b-n total (* 100.0 (/ b-n total))))
            (when (> n-samples 1)
              (println (format "Vote:       %d/%d (%.1f%%)" v-n total (* 100.0 (/ v-n total)))))

            (println)
            (println "By category:")
            (doseq [[cat rs] (sort-by key (group-by :source results))]
              (let [cat-b (count (filter :baseline rs))
                    cat-total (count rs)]
                (if (> n-samples 1)
                  (let [cat-v (count (filter :vote rs))]
                    (println (format "  %-30s: %2d/%2d baseline | %2d/%2d vote"
                                     cat cat-b cat-total cat-v cat-total)))
                  (println (format "  %-30s: %2d/%2d" cat cat-b cat-total)))))

            ;; Write results to JSON if requested
            (when out-file
              (spit out-file (json/write-str
                              {:timestamp (str (java.time.Instant/now))
                               :config {:n-questions total
                                        :n-samples n-samples
                                        :vote-temp vote-temp
                                        :category category}
                               :summary {:baseline b-n
                                         :total total
                                         :pct (float (* 100.0 (/ b-n total)))}
                               :results results}
                              :escape-slash false))
              (println (format "\nWrote results to %s" out-file)))))

        (finally
          (ds4/free-session session)
          (ds4/close-engine engine))))))

(defn -main [& args]
  ;; Parse CLI args
  (when (or (empty? args) (some #(= % "--help") args))
    (println "DS4 Eval Runner")
    (println)
    (println "Usage: clojure -M -m ds4-clj.eval-all [OPTIONS]")
    (println)
    (println "Options:")
    (println "  --category CAT     Filter to category (AIME2025, COMPSEC, GPQA Diamond, SuperGPQA)")
    (println "  --ids a,b,c        Run specific question IDs")
    (println "  --idxs 1,2,3       Run specific 1-based indices")
    (println "  --first-n N        Run only first N questions")
    (println "  --n-samples N      Enable majority voting with N samples (default: 1)")
    (println "  --vote-temp T      Temperature for voting samples (default: 0.7)")
    (println "  --tools            Enable Python calculator for math questions")
    (println "  --out-file FILE    Write results to JSON file")
    (println)
    (println "Examples:")
    (println "  clojure -M -m ds4-clj.eval-all --first-n 10")
    (println "  clojure -M -m ds4-clj.eval-all --category COMPSEC --out-file compsec.json")
    (println "  clojure -M -m ds4-clj.eval-all --n-samples 5 --vote-temp 0.8 --out-file voted.json")
    (println "  clojure -M -m ds4-clj.eval-all --tools --category AIME2025 --first-n 5")
    (println)
    (println "Generate report:")
    (println "  python3 tools/eval_report.py results.json > report.md")
    (println)
    (println "Compare runs:")
    (println "  python3 tools/diff_evals.py run1.json run2.json")
    (System/exit 0))

  (let [parsed (loop [[k v & more] args
                      acc {}]
                 (if-not k
                   acc
                   (condp = k
                     "--category" (recur more (assoc acc :category v))
                     "--ids" (recur more (assoc acc :ids (set (str/split v #","))))
                     "--idxs" (recur more (assoc acc :idxs (set (map #(Integer/parseInt %) (str/split v #",")))))
                     "--first-n" (recur more (assoc acc :first-n (Integer/parseInt v)))
                     "--n-samples" (recur more (assoc acc :n-samples (Integer/parseInt v)))
                     "--vote-temp" (recur more (assoc acc :vote-temp (Float/parseFloat v)))
                     "--out-file" (recur more (assoc acc :out-file v))
                     "--tools" (recur more (assoc acc :tools? true))
                     (do (println "Unknown arg:" k) (recur more acc)))))]
    (apply run-eval (mapcat identity parsed))))
