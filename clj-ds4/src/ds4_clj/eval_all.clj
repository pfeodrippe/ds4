(ns ds4-clj.eval-all
  "Full 92-question eval from Clojure REPL. Baseline only for speed comparison."
  (:require
   [clojure.string :as str]
   [clojure.data.json :as json]
   [ds4-clj.core :as ds4]))

(def questions
  (json/read-str (slurp "/tmp/eval_all_92.json") :key-fn keyword))

(defn check-answer
  "Match ds4_eval.c grading logic."
  [text expected choices]
  (let [visible (if (str/includes? text "</think>")
                  (second (str/split text #"</think>" 2))
                  text)
        vu (str/upper-case visible)]
    (if (and choices (= 1 (count expected)))
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
      (str/includes? visible expected))))

(defn run-with-model
  [model session q n-tokens]
  (let [text (ds4/generate-with-session model session (:prompt q)
                                        :n-tokens n-tokens :temperature 0.0)
        passed (check-answer text (:answer q) (:choices q))]
    [passed (count text)]))

(defn -main []
  (println (apply str (repeat 70 "=")))
  (println "FULL EVAL: 92 questions from Clojure REPL")
  (println (apply str (repeat 70 "=")))

  (println "\nLoading model...")
  (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal)
        base-model (ds4/make-model engine)
        session (ds4/create-session engine 4096)]
    (try
      (println "Model loaded!")
      (let [start (System/nanoTime)
            results (doall
                     (map-indexed
                      (fn [idx q]
                        (when (zero? (mod idx 10))
                          (println (format "\n--- Progress: %d/92 ---" idx)))
                        (let [[b-ok _] (run-with-model base-model session q 1500)]
                          (when b-ok
                            (print " PASS")
                            (flush))
                          {:idx (inc idx) :baseline b-ok}))
                      questions))
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
          (doseq [[cat rs] (sort-by key (group-by #(get-in questions [(dec (:idx %)) :source]) results))]
            (let [cat-b (count (filter :baseline rs))
                  cat-total (count rs)]
              (println (format "  %-30s: %2d/%2d" cat cat-b cat-total))))))

      (finally
        (ds4/free-session session)
        (ds4/close-engine engine)))))