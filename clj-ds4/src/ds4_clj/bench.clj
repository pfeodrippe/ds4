(ns ds4-clj.bench
  "Benchmark CLI vs REPL speed."
  (:require
   [clojure.string :as str]
   [ds4-clj.core :as ds4]))

(defn bench-repl
  "Run n generations from a loaded Model, timing each."
  [model prompt n-tokens n-runs]
  (println (format "REPL benchmark: %d runs, %d tokens each" n-runs n-tokens))
  (let [times (mapv (fn [i]
                      (let [start (System/nanoTime)
                            text (ds4/generate-model model prompt :n-tokens n-tokens :temperature 0.0)
                            elapsed (/ (- (System/nanoTime) start) 1e9)]
                        (println (format "  Run %d: %.3fs → %s"
                                         (inc i) elapsed
                                         (str/trim (subs text 0 (min 40 (count text))))))
                        elapsed))
                    (range n-runs))]
    (println (format "  Total: %.3fs, Avg: %.3fs/run"
                     (reduce + times)
                     (/ (reduce + times) n-runs)))))

(defn -main []
  (println "=" 60)
  (println "Loading model into REPL...")
  (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf" :backend :metal)
        model (ds4/make-model engine)]
    (try
      (println "Model loaded!")
      (println)

      ;; Benchmark 1: Simple question
      (bench-repl model "What is 2+2?" 200 3)
      (println)

      ;; Benchmark 2: Harder physics question
      (bench-repl model
        "Imagine an uncharged spherical conductor of radius R having a spherical cavity of radius r. A positive point charge +q is placed inside the cavity. Consider a point P outside the conductor at a distance L from the center. Which statement is correct? A) E depends on q, s, θ. B) E depends on q and r. C) E depends only on q and L. D) E depends on q, s, θ, and r."
        800 3)

      (println)
      (println "Done!")

      (finally
        (ds4/close-engine engine)))))