import "./App.scss";
import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState();
  const [submitted, setSubmitted] = useState(false);
  const [returnedData, setReturnedData] = useState([
    "none",
    "none",
    "none",
    "none",
  ]);

  const changeHandler = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmission = () => {
    const data = new FormData();
    data.append("file", selectedFile, selectedFile.name);
    axios({
      method: "post",
      url: "/upload",
      data: data,
    })
      .then((res) => {
        setSubmitted(true);
        console.log(res.data);
        setReturnedData([
          res.data.emotion,
          res.data.path,
          res.data.overall_emotion,
          [
            [res.data.precents.first.lab, res.data.precents.first.val],
            [res.data.precents.second.lab, res.data.precents.second.val],
            [res.data.precents.third.lab, res.data.precents.third.val],
          ],
        ]);
        console.log(returnedData);
      })
      .catch((err) => {
        throw err;
      });
  };

  /*useEffect(() => {
    fetch("/upload").then((data) => {
      returnedFile(data);
      console.log("a file was returned");
    });
  }, []);*/

  return (
    <div>
      <div className="header">EMOTION DETECTION</div>
      <div className="instructions">
        <ol>
          <li>Upload an image</li>
          <li>Ensure Correct File Format (jpg, jpeg, png)</li>
          <li>Submit File</li>
          <li>Recieve Results</li>
        </ol>
      </div>
      <div className="input">
        <input type="file" name="file" onChange={changeHandler} />
        <div>
          <button onClick={handleSubmission}>Submit</button>
        </div>
      </div>
      {submitted ? (
        <div className="result">
          <img src={returnedData.at(1)} alt="processed image" />
          <div className="answer">
            <div className="first_ans">
              <p className="lab">Overall:</p>
              <p className="ans">{returnedData.at(2)}</p>
            </div>
            <div className="first_ans">
              <p className="lab">Emotion Detected:</p>
              <p className="ans">{returnedData.at(0)}</p>
            </div>
            <div className="head_soft">
              <p>SoftMax: (Top 3)</p>
            </div>
            <div className="soft_ans">
              <p className="soft_lab">{returnedData.at(3).at(0).at(0)}</p>
              <p className="soft_ans">{returnedData.at(3).at(0).at(1)}</p>
            </div>
            <div className="soft_ans">
              <p className="soft_lab">{returnedData.at(3).at(1).at(0)}</p>
              <p className="soft_ans">{returnedData.at(3).at(1).at(1)}</p>
            </div>
            <div className="soft_ans">
              <p className="soft_lab">{returnedData.at(3).at(2).at(0)}</p>
              <p className="soft_ans">{returnedData.at(3).at(2).at(1)}</p>
            </div>
          </div>
        </div>
      ) : (
        <p></p>
      )}
    </div>
  );
}

export default App;
