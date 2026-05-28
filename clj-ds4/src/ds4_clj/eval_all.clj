(ns ds4-clj.eval-all
  "Full 92-question eval from Clojure REPL. Supports filtering by category, IDs, or indices."
  (:require
   [clojure.string :as str]
   [clojure.data.json :as json]
   [ds4-clj.core :as ds4]))

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

(defn run-with-model
  [model session q n-tokens]
  (let [text (ds4/generate-with-session model session (:prompt q)
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

  Examples:
    (run-eval)                          ; all 92 questions
    (run-eval :category \"COMPSEC\")     ; only COMPSEC
    (run-eval :ids #{\"compsec-077\" \"compsec-080\"}) ; specific questions
    (run-eval :idxs #{4 5 9})           ; questions #4, #5, #9
    (run-eval :first-n 10)              ; first 10 questions"
  [& {:keys [category ids idxs first-n]
      :or {first-n 92}}]
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
                          (let [[b-ok _] (run-with-model base-model session q 1500)]
                            (println (format "[%d/%d] #%2d %s: %s" (inc idx) (count filtered) (:idx q) (:id q) (if b-ok "PASS" "FAIL")))
                            {:idx (:idx q) :id (:id q) :baseline b-ok :source (:source q)}))
                        filtered))
              elapsed (/ (- (System/nanoTime) start) 1e9)]

          (println)
          (println (apply str (repeat 70 "=")))
          (println "SUMMARY")
          (println (apply str (repeat 70 "=")))
          (let [b-n (count (filter :baseline results))
                total (count results)]
            (println (format "Total time: %.1f minutes" (/ elapsed 60)))
            (println (format "Baseline:   %d/%d (%.1f%%)" b-n total (* 100.0 (/ b-n total))))

            (println)
            (println "By category:")
            (doseq [[cat rs] (sort-by key (group-by :source results))]
              (let [cat-b (count (filter :baseline rs))
                    cat-total (count rs)]
                (println (format "  %-30s: %2d/%2d" cat cat-b cat-total))))))

        (finally
          (ds4/free-session session)
          (ds4/close-engine engine))))))

(defn -main [& args]
  ;; Parse CLI args: --category X, --ids a,b,c, --idxs 4,5,9, --first-n 10
  (let [parsed (loop [[k v & more] args
                      acc {}]
                 (if-not k
                   acc
                   (condp = k
                     "--category" (recur more (assoc acc :category v))
                     "--ids" (recur more (assoc acc :ids (set (str/split v #","))))
                     "--idxs" (recur more (assoc acc :idxs (set (map #(Integer/parseInt %) (str/split v #",")))))
                     "--first-n" (recur more (assoc acc :first-n (Integer/parseInt v)))
                     (do (println "Unknown arg:" k) (recur more acc)))))]
    (apply run-eval (mapcat identity parsed))))
