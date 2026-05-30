(ns ds4-clj.lora-test
  "Test LoRA stub API.

  Run with:
    cd clj-ds4 && DS4_TEST_MODEL=qwen3-coder.gguf clojure -M:test -n ds4-clj.lora-test"
  (:require
   [clojure.test :refer [deftest is testing use-fixtures]]
   [ds4-clj.core :as ds4]))

(def ^:dynamic *engine* nil)

(defn engine-fixture
  [f]
  (let [engine (ds4/open-engine :model-path "qwen3-coder.gguf"
                                :backend :metal)]
    (binding [*engine* engine]
      (try
        (f)
        (finally
          (ds4/close-engine engine))))))

(use-fixtures :once engine-fixture)

(deftest test-lora-stub-api
  (testing "LoRA stub API compiles and returns expected values"
    ;; Initially not enabled
    (is (not (ds4/lora-enabled? *engine*))
        "LoRA should not be enabled initially")

    ;; Initialize
    (ds4/lora-init! *engine* :rank 8 :alpha 16 :lr 1e-4)
    (is (ds4/lora-enabled? *engine*)
        "LoRA should be enabled after init")

    ;; Free
    (ds4/lora-free! *engine*)
    (is (not (ds4/lora-enabled? *engine*))
        "LoRA should not be enabled after free")

    ;; Re-init with defaults
    (ds4/lora-init! *engine*)
    (is (ds4/lora-enabled? *engine*))

    ;; Save/load stubs return errors (expected — not implemented)
    (is (thrown? Exception (ds4/lora-save! *engine* "/tmp/test_lora.bin"))
        "Save stub should throw (not implemented)")
    (is (thrown? Exception (ds4/lora-load! *engine* "/tmp/test_lora.bin"))
        "Load stub should throw (not implemented)")

    ;; Clean up
    (ds4/lora-free! *engine*)))
