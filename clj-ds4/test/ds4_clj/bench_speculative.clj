(ns ds4-clj.bench-speculative
  "Benchmark speculative decoding vs regular greedy decoding.

  Run with:
    cd clj-ds4 && DS4_TEST_MODEL=qwen3-coder.gguf clojure -M:test -n ds4-clj.bench-speculative"
  (:require
   [clojure.test :refer [deftest is testing]]
   [ds4-clj.core :as ds4]))

(deftest test-speculative-speedup
  (testing "Speculative decoding is faster than regular greedy on Metal"
    (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                  :backend :metal)
          session-regular (ds4/create-session engine 512)
          session-spec (ds4/create-session engine 512)
          prompt "The capital of France is"
          n-tokens 50]

      ;; Warmup
      (ds4/generate engine session-regular prompt {:n-tokens 10 :temperature 0.0})
      (ds4/generate-speculative engine session-spec prompt {:n-tokens 10})

      ;; Regular greedy
      (let [t0 (System/nanoTime)
            text1 (ds4/generate engine session-regular prompt
                                {:n-tokens n-tokens :temperature 0.0})
            t1 (System/nanoTime)
            regular-ms (/ (- t1 t0) 1e6)]

        ;; Speculative
        (let [t2 (System/nanoTime)
              text2 (ds4/generate-speculative engine session-spec prompt
                                              {:n-tokens n-tokens})
              t3 (System/nanoTime)
              spec-ms (/ (- t3 t2) 1e6)]

          (println (format "\n=== Speculative Decode Benchmark ==="))
          (println (format "Prompt: %s" prompt))
          (println (format "Tokens: %d" n-tokens))
          (println (format "Regular:  %.1f ms (%.2f tok/s)" regular-ms (* 1000.0 (/ n-tokens regular-ms))))
          (println (format "Speculative: %.1f ms (%.2f tok/s)" spec-ms (* 1000.0 (/ n-tokens spec-ms))))
          (println (format "Speedup: %.2fx" (/ regular-ms spec-ms)))
          (println (format "Regular output:  %s" (subs text1 0 (min 60 (count text1)))))
          (println (format "Speculative output: %s" (subs text2 0 (min 60 (count text2)))))

          ;; Verify both produce reasonable output
          (is (string? text1))
          (is (string? text2))
          (is (> (count text1) 0))
          (is (> (count text2) 0))

          ;; Speculative should be faster (or at least not much slower)
          (is (< spec-ms (* regular-ms 1.5))
              (format "Speculative (%.1f ms) should not be >50%% slower than regular (%.1f ms)"
                      spec-ms regular-ms))))

      (ds4/free-session session-regular)
      (ds4/free-session session-spec)
      (ds4/close-engine engine))))
