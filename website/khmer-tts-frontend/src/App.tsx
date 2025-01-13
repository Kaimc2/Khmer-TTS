import { useState } from "react";
import logo from "./assets/logo.png";
import loadingIcon from "./assets/loading.png";
import axios from "axios";
import "./App.css";

const backend_url = "http://127.0.0.1:5000";

function App() {
  const [text, setText] = useState("");
  const [audioSrc, setAudioSrc] = useState("");
  const [loading, setLoading] = useState(false);

  const synthesizeSpeech = () => {
    console.log("Synthesizing speech for:", text);
    setLoading(true);

    axios
      .post(backend_url + "/synthesize", { text }, { responseType: "blob" })
      .then((response) => {
        console.log(response);
        setLoading(false);
        const audioBlob = response.data;
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioSrc(audioUrl);
      })
      .catch((error) => {
        setLoading(false);
        console.error("Error synthesize speech:", error);
      });
  };

  return (
    <>
      <div>
        <img src={logo} className="logo" alt="Logo" />
      </div>
      <h1>Veaja - វាចា (Khmer TTS)</h1>
      <div className="card">
        <div className="container">
          <input
            className="input-field"
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button onClick={synthesizeSpeech}>Synthesize</button>
        </div>
      </div>
      {loading && <img className="loader" src={loadingIcon} alt="loading" />}
      {audioSrc && (
        <div>
          <p>Text: {text}</p>
          <audio controls>
            <source src={audioSrc} type="audio/wav" />
            Your browser does not support the audio element.
          </audio>
        </div>
      )}
    </>
  );
}

export default App;
