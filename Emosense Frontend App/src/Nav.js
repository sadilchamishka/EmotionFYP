
import React from 'react';
import './Nav.css';
import {Link} from 'react-router-dom';

function Nav() {

    const navStyle = {
        color:'white'
    };

    return (
        <nav>
            <ul className="nav-links">
                <Link style={navStyle} to="/uploadutterence"> Upload Utterence</Link> 
                <Link style={navStyle} to="/recordutterence"> Record Utterence </Link>
                <Link style={navStyle} to="/uploadconversation"> Upload Conversation </Link>
            </ul>
        </nav>
    )
}

export default Nav;