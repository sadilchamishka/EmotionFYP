import React, {useState } from 'react';
import {emotionProbDict} from './Util'

//********************************//
import RadarChart from 'react-svg-radar-chart';
import 'react-svg-radar-chart/build/css/index.css'
//*********************************//

import {Button,Input} from '@material-ui/core';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import { green } from '@material-ui/core/colors';
import { withStyles,makeStyles } from '@material-ui/core/styles';
import {
  Chart,
  BarSeries,
  Title,
  ArgumentAxis,
  ValueAxis,
} from '@devexpress/dx-react-chart-material-ui';

import { Animation } from '@devexpress/dx-react-chart';
import plutchik from 'plutchik.png';

const serverURL = "http://localhost:5000/";

//*********************************//
const captions = {
      // columns
      happy: 'Happy',
      excited: 'Excited',
      sad: 'Sad',      
      frustration: 'Frustration',
      angry: 'Angry',
    };
//*********************************//
  
const useStyles = makeStyles((theme) => ({
    paper1: {
      maxWidth: 730,
      margin: `${theme.spacing(1)}px auto`,
      padding: theme.spacing(2),
      textAlign: 'left',
      color: theme.palette.text.secondary,
    }
  }));

const ColorButton = withStyles((theme) => ({
    root: {
      color: theme.palette.getContrastText(green[700]),
      backgroundColor: green[400],
      '&:hover': {
        backgroundColor: green[800],
      },
    },
  }))(Button);

function UploadUtterence() {

const [predictions, setPredictions] = useState(emotionProbDict(0));

//*********************************//
const [radarChartData, setRadarChartData] = useState([{ data: {  happy: 0,
			  					excited: 0,
								sad: 0,
								angry: 0,
								frustration: 0 } } ]);
//*********************************//

const classes = useStyles();

const submitUtterence = async () =>{
    let file = document.getElementById("f1").files[0];
    let formData = new FormData();
    formData.append("audio", file);
    const response = await fetch(serverURL.concat("utterence"), {method: "POST", body: formData});
    const data = await response.json();
    
    const responsePredictions = emotionProbDict(data);
    setPredictions(responsePredictions);
    
    //*********************************//
    const data_ = data["prediction"];

    setRadarChartData([{ data: {  happy: data_[0],
			           excited: data_[4],
			           sad: data_[1],
			           angry: data_[3],
			           frustration: data_[5]}, meta: {color: 'blue'} } ]);
    //*********************************//
}

  return (
    <div>
    <br></br>
    <Grid>
      <Paper className={classes.paper1}>
          <Input type="file" id="f1" color="primary"></Input>  &emsp;
          <ColorButton style={{float: 'right'}}  onClick={submitUtterence} variant="contained" color="primary"> Predict </ColorButton>
         
      </Paper>

    </Grid>

    
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
       
   <img src={plutchik} width='670' height='500'/>

    
    <RadarChart captions={captions} data={radarChartData} size={500} />


  </div>
  );
}

export default UploadUtterence;
