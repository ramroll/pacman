import React, {Component} from 'app'
import ReactDOM from 'react-dom'

const Map = ({ state }) => <table>

  <tbody>

    {state.map((row, i) => {
      return <tr key={i}>
        {row.map((col, j) => {
          const x = col[j]
          let cell = null
          let cls = ''

          if(col[j] === 1) {
            cls = 'wall'
          }

          switch(col[j]) {
            case 2:
              cell = <div className='food'>.</div>
            case 3:
              cell = <div className='packman'>P</div>
            case 5:
              cell = <div className='ghost'>G</div> 
          }

          return <td className={cls} key={j}>
            {cell}
          </td>
        })}


      </tr>
    })}

  </tbody>
</table>







