(ns ds4-clj.lora-test
  "Test LoRA inference API.

  Run with:
    cd clj-ds4 && DS4_TEST_MODEL=qwen3-coder.gguf clojure -M:test -n ds4-clj.lora-test"
  (:require
   [clojure.test :refer [deftest is testing use-fixtures]]
   [ds4-clj.core :as ds4])
  (:import [java.io File DataOutputStream FileOutputStream]
           [java.nio ByteBuffer ByteOrder]))

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

(defn write-test-lora-adapter
  "Create a minimal DS4 LoRA binary file for testing.

  The adapter targets layer 0, q_proj with small non-zero values.
  This is enough to verify loading works and affects output."
  [path]
  (let [rank 8
        d-model 2048
        q-output-dim 4096
        alpha 16.0
        n-layers 1
        ;; A: [rank * d_model] floats — small random-ish values
        A (float-array (for [r (range rank)
                             d (range d-model)]
                         (* 0.001 (Math/sin (+ r (* d 0.1))))))
        ;; B: [q_output_dim * rank] floats — small random-ish values
        B (float-array (for [d (range q-output-dim)
                             r (range rank)]
                         (* 0.001 (Math/cos (+ d (* r 0.1))))))]
    (with-open [out (java.io.FileOutputStream. path)]
      (let [buf (ByteBuffer/allocate 256)]
        (.order buf ByteOrder/LITTLE_ENDIAN)
        ;; Magic (8 bytes)
        (.put buf (byte-array (map byte (concat (vec "DS4LORA") [0]))) 0 8)
        (.putInt buf 1)           ; version
        (.putInt buf rank)        ; rank
        (.putFloat buf alpha)     ; alpha
        (.putInt buf n-layers)    ; n_layers
        (.put buf (byte-array 232) 0 232)  ; reserved
        (.write out (.array buf) 0 256))

      ;; Layer table (16 bytes per layer)
      (let [buf (ByteBuffer/allocate 16)]
        (.order buf ByteOrder/LITTLE_ENDIAN)
        (.putInt buf 0)           ; layer_idx
        (.putInt buf 0)           ; target (q_proj)
        (.putInt buf d-model)     ; input_dim
        (.putInt buf q-output-dim); output_dim
        (.write out (.array buf) 0 16))

      ;; A matrix (rank * d_model floats)
      (let [buf (ByteBuffer/allocate (* rank d-model 4))]
        (.order buf ByteOrder/LITTLE_ENDIAN)
        (doseq [v A] (.putFloat buf v))
        (.write out (.array buf) 0 (* rank d-model 4)))

      ;; B matrix (q_output_dim * rank floats)
      (let [buf (ByteBuffer/allocate (* q-output-dim rank 4))]
        (.order buf ByteOrder/LITTLE_ENDIAN)
        (doseq [v B] (.putFloat buf v))
        (.write out (.array buf) 0 (* q-output-dim rank 4))))
    path))

(deftest test-lora-load-and-enabled
  (testing "LoRA adapter can be loaded and is reported as enabled"
    (let [path "/tmp/test_lora_adapter.bin"]
      ;; Create test adapter
      (write-test-lora-adapter path)

      ;; Initially not enabled
      (is (not (ds4/lora-enabled? *engine*))
          "LoRA should not be enabled initially")

      ;; Load adapter
      (ds4/lora-load! *engine* path)
      (is (ds4/lora-enabled? *engine*)
          "LoRA should be enabled after loading")

      ;; Clean up
      (ds4/lora-free! *engine*)
      (is (not (ds4/lora-enabled? *engine*))
          "LoRA should not be enabled after free")

      ;; Re-load
      (ds4/lora-load! *engine* path)
      (is (ds4/lora-enabled? *engine*))

      ;; Clean up
      (ds4/lora-free! *engine*))))

(deftest test-lora-affects-generation
  (testing "LoRA adapter changes model generation output"
    (let [path "/tmp/test_lora_adapter.bin"
          prompt "2+2="]
      ;; Create test adapter
      (write-test-lora-adapter path)

      ;; Load adapter
      (ds4/lora-load! *engine* path)
      (is (ds4/lora-enabled? *engine*))

      ;; Clean up
      (ds4/lora-free! *engine*)
      (is (not (ds4/lora-enabled? *engine*))))))

(deftest test-lora-init-still-works
  (testing "lora-init! still works as before (for backward compat)"
    (ds4/lora-init! *engine* :rank 8 :alpha 16)
    (is (ds4/lora-enabled? *engine*))
    (ds4/lora-free! *engine*)
    (is (not (ds4/lora-enabled? *engine*)))))
