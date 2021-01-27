import React, {useState } from 'react';

import {Button,Input} from '@material-ui/core';
import Paper from '@material-ui/core/Paper';
import Grid from '@material-ui/core/Grid';
import { green } from '@material-ui/core/colors';
import { withStyles,makeStyles } from '@material-ui/core/styles';
import { Animation } from '@devexpress/dx-react-chart';
import {
    Chart,
    ArgumentAxis,
    ValueAxis,
    LineSeries,
    Title,
    Legend,
  } from '@devexpress/dx-react-chart-material-ui';

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

function UploadConversation() {

const [maleChartData, setMaleChartData] = useState([]);
const [femaleChartData, setFemaleChartData] = useState([]);
const [attentionChartData, setAttentionChartData] = useState([]);

const [speakers, setSpeakers] = useState("");
const classes = useStyles();

const submitUtterence = async () =>{
    let file = document.getElementById("f1").files;
    let formData = new FormData();

    for (var index = 0; index < file.length; index++) { 
        formData.append(index.toString(), file[index]);
    } 
    
    var queryParam = "?speakers=";
    console.log(speakers);
    queryParam = queryParam.concat(speakers);
    const response = await fetch(serverURL.concat("conversation/offline").concat(queryParam), {method: "POST", body: formData});
    const data = await response.json();

    var sp = speakers.split(',');
    var maleSpeaker = [];
    var femaleSpeaker = [];
    console.log(data);
    for (var index = 0; index < data['prediction'].length; index++) { 
        if (sp[index]=='M'){
            maleSpeaker.push(data['prediction'][index]);
        } else {
            femaleSpeaker.push(data['prediction'][index]);
        }
    } 
    setMaleChartData(maleSpeaker);
    setFemaleChartData(femaleSpeaker);
    setAttentionChartData(data.attention);
    
}

const format = () => tick => tick;
const legendStyles = () => ({
  root: {
    display: 'flex',
    margin: 'auto',
    flexDirection: 'row',
  },
});
const legendLabelStyles = theme => ({
  label: {
    paddingTop: theme.spacing(1),
    whiteSpace: 'nowrap',
  },
});
const legendItemStyles = () => ({
  item: {
    flexDirection: 'column',
  },
});

const legendRootBase = ({ classes, ...restProps }) => (
  <Legend.Root {...restProps} className={classes.root} />
);
const legendLabelBase = ({ classes, ...restProps }) => (
  <Legend.Label className={classes.label} {...restProps} />
);
const legendItemBase = ({ classes, ...restProps }) => (
  <Legend.Item className={classes.item} {...restProps} />
);
const Root = withStyles(legendStyles, { name: 'LegendRoot' })(legendRootBase);
const Label = withStyles(legendLabelStyles, { name: 'LegendLabel' })(legendLabelBase);
const Item = withStyles(legendItemStyles, { name: 'LegendItem' })(legendItemBase);

const ValueLabel = (props) => {
  const { text } = props;
  return (
    <ValueAxis.Label
      {...props}
      text={`${text}%`}
    />
  );
};

const titleStyles = {
  title: {
    whiteSpace: 'pre',
  },
};
const TitleText = withStyles(titleStyles)(({ classes, ...props }) => (
  <Title.Text {...props} className={classes.title} />
));

const addSpeaker = (event)=>{
    setSpeakers(event.target.value);
}

  return (
    <div>
    <br></br>
    <Grid>
      <Paper className={classes.paper1}>
          <input type="file" id="f1" color="primary" multiple/> &emsp;
          <input type="text" value={speakers} onChange={addSpeaker}/>
          <ColorButton style={{float: 'right'}}  onClick={submitUtterence} variant="contained" color="primary"> Predict </ColorButton>
          <br></br>
      </Paper>
    </Grid>
    <br></br><br></br>

    <Paper>
        <Chart
          data={maleChartData}
          className={classes.chart}
        >
          <ArgumentAxis tickFormat={format} />
          <ValueAxis
            max={50}
            labelComponent={ValueLabel}
          />

          <LineSeries
            name="Happy"
            valueField="Happy"
            argumentField="timestep"
          />
          <LineSeries
            name="Sad"
            valueField="Sad"
            argumentField="timestep"
          />
          <LineSeries
            name="Neutral"
            valueField="Neutral"
            argumentField="timestep"
          />
           <LineSeries
            name="Angry"
            valueField="Angry"
            argumentField="timestep"
          />
           <LineSeries
            name="Excited"
            valueField="Excited"
            argumentField="timestep"
          />
           <LineSeries
            name="Frustrated"
            valueField="Frustrated"
            argumentField="timestep"
          />
          <Legend position="bottom" rootComponent={Root} itemComponent={Item} labelComponent={Label} />
          <Title
            text="Emotion Variation of First Speaker"
            textComponent={TitleText}
          />
          <Animation />
        </Chart>
      </Paper>


    <br></br><br></br>
    <Paper>
        <Chart
          data={femaleChartData}
          className={classes.chart}
        >
          <ArgumentAxis tickFormat={format} />
          <ValueAxis
            max={50}
            labelComponent={ValueLabel}
          />

          <LineSeries
            name="Happy"
            valueField="Happy"
            argumentField="timestep"
          />
          <LineSeries
            name="Sad"
            valueField="Sad"
            argumentField="timestep"
          />
          <LineSeries
            name="Neutral"
            valueField="Neutral"
            argumentField="timestep"
          />
           <LineSeries
            name="Angry"
            valueField="Angry"
            argumentField="timestep"
          />
           <LineSeries
            name="Excited"
            valueField="Excited"
            argumentField="timestep"
          />
           <LineSeries
            name="Frustrated"
            valueField="Frustrated"
            argumentField="timestep"
          />
          <Legend position="bottom" rootComponent={Root} itemComponent={Item} labelComponent={Label} />
          <Title
            text="Emotion Variation of Second Speaker"
            textComponent={TitleText}
          />
          <Animation />
        </Chart>
      </Paper>

      <br></br><br></br>
    <Paper>
        <Chart
          data={attentionChartData}
          className={classes.chart}
          >
          <ArgumentAxis tickFormat={format} />
          <ValueAxis
            max={50}
            labelComponent={ValueLabel}
          />
          <LineSeries
            name="Forward"
            valueField="Forward"
            argumentField="timestep"
          />
          <LineSeries
            name="Backward"
            valueField="Backward"
            argumentField="timestep"
          /> 
          <Legend position="bottom" rootComponent={Root} itemComponent={Item} labelComponent={Label} />
          <Title
            text="Emotional Attention over the Conversation"
            textComponent={TitleText}
          />
          <Animation />
        </Chart>
      </Paper>
  </div>
  );
}

export default UploadConversation;


