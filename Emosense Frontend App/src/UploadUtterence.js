import React, {useState } from 'react';
import {emotionProbDict} from './Util'

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

const serverURL = "http://localhost:5000/";

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
const classes = useStyles();

const submitUtterence = async () =>{
    let file = document.getElementById("f1").files[0];
    let formData = new FormData();
    formData.append("audio", file);
    const response = await fetch(serverURL.concat("utterence"), {method: "POST", body: formData});
    const data = await response.json();
    
    const responsePredictions = emotionProbDict(data);
    setPredictions(responsePredictions);
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
  </div>
  );
}

export default UploadUtterence;
