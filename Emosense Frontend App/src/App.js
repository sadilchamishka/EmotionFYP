import {BrowserRouter as Router, Switch, Route} from 'react-router-dom';
import './App.css';
import Nav from './Nav';
import UploadUtterence from './UploadUtterence';
import RecordUtterence from './RecordUtterence';
import UploadConversation from './UploadConversation';

function App() {

  return (
    <div className="App">
      <h1>EMOSENSE</h1>
       <Router>
            <Nav/>
              <Switch>
                <Route path="/uploadutterence" component={UploadUtterence}/>
                <Route path="/recordutterence" component={RecordUtterence}/>
                <Route path="/uploadconversation" component={UploadConversation}/>
              </Switch>
        </Router>
    </div>
  );
}

export default App;
