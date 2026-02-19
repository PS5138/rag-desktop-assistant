import { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setResponse("");

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: query }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      setResponse(data.answer);
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>📁 RAG Desktop Assistant</h1>
      <textarea
        rows={4}
        placeholder="Ask a question about your documents..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Thinking..." : "Submit"}
      </button>

      {response && (
        <div className="response">
          <strong>Answer:</strong>
          <p>{response}</p>
        </div>
      )}

      {error && <p className="error">❌ {error}</p>}
    </div>
  );
}

export default App;