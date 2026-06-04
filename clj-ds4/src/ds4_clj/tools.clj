(ns ds4-clj.tools
  "Tool-augmented generation for DS4.

  The model can emit tool calls in a simple format:
  <tool>python</tool>
  <code>
  import math
  math.factorial(10)
  </code>

  The tool executes the code and returns the result, which the model
  can use in subsequent reasoning."
  (:require
   [clojure.string :as str]
   [clojure.java.shell :as shell]
   [ds4-clj.core :as ds4]))

(def ^:private python-eval-wrapper
  (str "import ast, pathlib, sys\n"
       "path = pathlib.Path(sys.argv[1])\n"
       "source = path.read_text()\n"
       "tree = ast.parse(source, filename=str(path))\n"
       "scope = {}\n"
       "if tree.body and isinstance(tree.body[-1], ast.Expr):\n"
       "    expr = tree.body.pop()\n"
       "    exec(compile(tree, str(path), 'exec'), scope, scope)\n"
       "    value = eval(compile(ast.fix_missing_locations(ast.Expression(expr.value)), str(path), 'eval'), scope, scope)\n"
       "    if value is not None:\n"
       "        print(value)\n"
       "else:\n"
       "    exec(compile(tree, str(path), 'exec'), scope, scope)\n"))

(defn- python-eval
  "Execute Python code in a subprocess and return stdout.

  Like a REPL, a final expression is printed even without an explicit print()."
  [code]
  (let [temp-file (java.io.File/createTempFile "ds4_tool_" ".py")]
    (try
      (spit temp-file code)
      (let [result (shell/sh "python3" "-c" python-eval-wrapper
                             (.getAbsolutePath temp-file))]
        (if (zero? (:exit result))
          (str/trim (:out result))
          (str "ERROR: " (str/trim (:err result)))))
      (finally
        (.delete temp-file)))))

(defn extract-tool-calls
  "Extract tool calls from model output.

  Returns a sequence of {:tool name :code code} maps,
  or nil if no tool calls found."
  [text]
  (let [pattern #"<tool>(\w+)</tool>\s*<code>\s*([\s\S]*?)\s*</code>"
        matches (re-seq pattern text)]
    (when (seq matches)
      (for [[_ tool code] matches]
        {:tool tool :code (str/trim code)}))))

(defn execute-tool
  "Execute a single tool call and return the result string."
  [{:keys [tool code]}]
  (case tool
    "python" (python-eval code)
    (str "Unknown tool: " tool)))

(defn execute-tools
  "Execute all tool calls in text and return a map of results.

  Returns {:results [{:tool :code :result}] :has-tools? bool}"
  [text]
  (if-let [calls (extract-tool-calls text)]
    {:results (for [call calls]
                (assoc call :result (execute-tool call)))
     :has-tools? true}
    {:results [] :has-tools? false}))

(defn- as-model
  [engine-or-model]
  (if (and (instance? clojure.lang.ILookup engine-or-model)
           (:engine engine-or-model))
    engine-or-model
    (ds4/make-model engine-or-model)))

(defn format-tool-results
  "Format tool results for feeding back to the model."
  [results]
  (str/join "\n\n"
    (for [{:keys [tool code result]} results]
      (format "<tool_result>\nTool: %s\nCode:\n%s\nResult: %s\n</tool_result>"
              tool code result))))

(defn generate-with-tools
  "Generate text with tool augmentation.

  The model is given a system prompt that encourages tool use.
  If the model emits tool calls, they are executed and the results
  are fed back. The model then continues generation.

  Options:
    :max-tool-rounds  - max tool rounds (default: 3)
    :n-tokens         - max tokens per generation (default: 1500)
    :temperature      - sampling temperature (default: 0.0)"
  [engine-or-model session prompt & {:keys [max-tool-rounds n-tokens temperature]
                                      :or {max-tool-rounds 3 n-tokens 1500 temperature 0.0}}]
  (let [tool-system (str "You have access to a Python calculator tool. "
                         "When you need to compute something, use:\n"
                         "<tool>python</tool>\n"
                         "<code>\n"
                         "# your python code here\n"
                         "</code>\n\n"
                         "The result will be provided to you. "
                         "After receiving a tool result, answer using it instead "
                         "of repeating the same tool call.")
        model (as-model engine-or-model)
        model-with-tools (ds4/with-system model tool-system)]
    (loop [round 0
           text ""
           history []]
      (let [full-prompt (if (seq history)
                          (str prompt "\n\n" (str/join "\n\n" history))
                          prompt)
            chunk (ds4/generate-with-session model-with-tools session full-prompt
                                                      :n-tokens n-tokens :temperature temperature)
            new-text (str text chunk)
            {:keys [results has-tools?]} (execute-tools chunk)]
        (if (and has-tools? (< round max-tool-rounds))
          (let [tool-output (format-tool-results results)]
            (recur (inc round)
                   (str new-text "\n\n" tool-output "\n\n")
                   (conj history (str chunk "\n\n" tool-output))))
          new-text)))))
