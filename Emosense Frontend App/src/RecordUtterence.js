import React, { useEffect, useState } from 'react';
import Paper from '@material-ui/core/Paper';

import {generateWaveFile,emotionProbDict} from './Util'
import {
    Chart,
    BarSeries,
    Title,
    ArgumentAxis,
    ValueAxis,
  } from '@devexpress/dx-react-chart-material-ui';

import './RecordUtterence.css';
import { Animation } from '@devexpress/dx-react-chart';

var leftchannel = [];
var recorder = null;
var recordingLength = 0;
var mediaStream = null;
var context = null; 
var serverURL = "http://localhost:5000/";

function RecordUtterence() {

const [recordButtonStyle, setRecordButtonStyle] = useState('notRec');
const [predictions, setPredictions] = useState(emotionProbDict(0));

const sendWave = async (formData) => {
    var response = await fetch(serverURL.concat("utterence"), {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    const prob = emotionProbDict(data);
    setPredictions(prob);
}

const recordAudio = () => {

  if (recordButtonStyle=='notRec'){

    setRecordButtonStyle('Rec');
    // Initialize recorder
    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
    navigator.getUserMedia(
    {
        audio: true
    },
    function (e) {
        console.log("Conversation is Recording");

        // creates the audio context
        window.AudioContext = window.AudioContext || window.webkitAudioContext;
        context = new AudioContext();
        mediaStream = context.createMediaStreamSource(e);  // creates an audio node from the microphone incoming stream

        // bufferSize: the onaudioprocess event is called when the buffer is full
        var bufferSize = 2048;
        var numberOfInputChannels = 2;   // Two channels, but one channel is used to reduce the size of audio file
        var numberOfOutputChannels = 2;

        if (context.createScriptProcessor) {
            recorder = context.createScriptProcessor(bufferSize, numberOfInputChannels, numberOfOutputChannels);
        } else {
            recorder = context.createJavaScriptNode(bufferSize, numberOfInputChannels, numberOfOutputChannels);
        }

        recorder.onaudioprocess = function (e) {
            leftchannel.push(new Float32Array(e.inputBuffer.getChannelData(0)));
            console.log("BUFFERING");
            recordingLength += bufferSize;
        }

        // we connect the recorder
        mediaStream.connect(recorder);
        recorder.connect(context.destination);
    },
    function (e) {
        console.error(e);
        });

  } else{
    setRecordButtonStyle('notRec');
    recorder.disconnect(context.destination);
    mediaStream.disconnect(recorder);
    sendWave(generateWaveFile(leftchannel,recordingLength));
  }
  
}
  return (
    <div>
        <br></br><br></br>    
        <button id="recButton" onClick={recordAudio} className={recordButtonStyle}></button>
        <p>Click the button to start/stop recording</p> 
              
        <Paper>
          <Chart data={predictions}>
            <ArgumentAxis />
            <ValueAxis max={7} />

            <BarSeries
              valueField="probability"
              argumentField="emotion"
            />
            <Title text="Emotion Prediction" />
            <Animation />
          </Chart>
        </Paper>
    
    </div>
  );
}

export default RecordUtterence;