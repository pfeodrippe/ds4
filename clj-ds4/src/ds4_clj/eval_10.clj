(ns ds4-clj.eval-10
  "Fast eval of 10 questions from REPL (model loaded once)."
  (:require
   [clojure.string :as str]
   [clojure.data.json :as json]
   [ds4-clj.core :as ds4]))

(def questions
  (json/read-str (slurp "/tmp/selected_10_failures.json") :key-fn keyword))

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
  "Run one question with a Model and session, return [passed text]."
  [model session q n-tokens]
  (let [text (ds4/generate-with-session model session (:prompt q)
                                        :n-tokens n-tokens :temperature 0.0)
        passed (check-answer text (:answer q) (:choices q))]
    [passed text]))

(defn -main []
  (println "=" 70)
  (println "REPL EVAL: 10 questions, model loaded once")
  (println "=" 70)

  (println "\nLoading model...")
  (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal)
        base-model (ds4/make-model engine)
        session (ds4/create-session engine 4096)]
    (try
      (println "Model loaded!")
      (let [start (System/nanoTime)
            results
            (doall
             (map-indexed
              (fn [idx q]
                (println)
                (println (format "#%2d: %s (%s) → %s"
                                (:idx q) (:id q) (:domain q) (:answer q)))

                ;; Baseline
                (print "  Baseline... ")
                (flush)
                (let [[b-ok b-text] (run-with-model base-model session q 1500)]
                  (println (if b-ok "PASS" "FAIL"))

                  (if b-ok
                    {:idx (:idx q) :baseline true :think false :think-max false}

                    ;; Think
                    (do
                      (print "  +think... ")
                      (flush)
                      (let [[t-ok t-text] (run-with-model (ds4/with-think base-model :normal) session q 1500)]
                        (println (if t-ok "PASS" "FAIL"))

                        ;; Think-max
                        (print "  +think-max... ")
                        (flush)
                        (let [[tm-ok tm-text] (run-with-model (ds4/with-think base-model :max) session q 1500)]
                          (println (if tm-ok "PASS" "FAIL"))

                          (when (and tm-ok (not b-ok))
                            (println "  *** FIXED ***"))

                          {:idx (:idx q) :baseline b-ok :think t-ok :think-max tm-ok}))))))
              questions))
            elapsed (/ (- (System/nanoTime) start) 1e9)]

        (println)
        (println (str "=" 70))
        (println "SUMMARY")
        (println (str "=" 70))
        (let [b-n (count (filter :baseline results))
              t-n (count (filter :think results))
              tm-n (count (filter :think-max results))]
          (println (format "Total time: %.1f minutes" (/ elapsed 60)))
          (println (format "Baseline:   %d/10" b-n))
          (println (format "+think:     %d/10" t-n))
          (println (format "+think-max: %d/10" tm-n))
          (println (format "Fixed:      %d" (- tm-n b-n)))

          (println)
          (println "Per-question:")
          (doseq [r results]
            (let [best (cond (:think-max r) "think-max"
                            (:think r) "think"
                            (:baseline r) "baseline"
                            :else "none")]
              (println (format "  #%2d: base=%5s think=%5s think-max=%5s → %s"
                              (:idx r) (:baseline r) (:think r) (:think-max r) best))))))

      (finally
        (ds4/free-session session)
        (ds4/close-engine engine)))))