import React, {useState } from 'react';
import {distressProbDict} from './Util'

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

const serverURL = "http://localhost:5001/";

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

function UploadSpeech() {

const [predictions, setPredictions] = useState(distressProbDict(1000));
const classes = useStyles();

const submitSpeech = async () =>{
  
  
    let file = document.getElementById("f1").files[0];
    let formData = new FormData();
    formData.append("audio", file);
    const response = await fetch(serverURL.concat("distressProbability"), {method: "POST", body: formData});
    const data = await response.json();
   
    const responsePredictions = distressProbDict(data);
    setPredictions(responsePredictions);
}

  return (
    <div>
    <br></br>
    <Grid>
      <Paper className={classes.paper1}>
          <Input type="file" id="f1" color="primary"></Input>  &emsp;
          <ColorButton style={{float: 'right'}}  onClick={submitSpeech} variant="contained" color="primary"> Detect </ColorButton>
      </Paper>
    </Grid>

     <Paper>
      <Chart data={predictions}>
        <ArgumentAxis />
        <ValueAxis max={3} />

        <BarSeries
          valueField="probability"
          argumentField="status"
        />
        <Title text="Distress Detection" />
        <Animation />
      </Chart>
    </Paper>

    
  </div>
  );
}

export default UploadSpeech;
