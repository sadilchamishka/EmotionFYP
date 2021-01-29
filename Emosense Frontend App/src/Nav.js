
import React from 'react';
import './Nav.css';
import {Link} from 'react-router-dom';

function Nav() {

    const navStyle = {
        color:'white'
    };

    return (
        <div>
            <nav1>

             <div className="split1 left">Detect Distress</div>
              <div className="split2 right">Predict Emotions</div>
           

            </nav1>
            <nav>
                <ul className="nav-links">
                    <Link style={navStyle} to="/uploaddistress"> Upload Speech</Link> 
                    <Link style={navStyle} to="/recorddistress"> Record Speech</Link> 
                    <Link style={navStyle} to="/uploadutterence"> Upload Utterence</Link> 
                    <Link style={navStyle} to="/recordutterence"> Record Utterence </Link>
                    <Link style={navStyle} to="/uploadconversation"> Upload Conversation </Link>
                </ul>
            </nav>
        </div>
    )
}

export default Nav;