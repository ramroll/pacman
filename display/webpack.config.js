const path = require('path')
const HtmlWebpackPlugin = require('html-webpack-plugin')
module.exports = {
  entry: './index.jsx',
  mode : 'development',
  devtool : 'source-map',
  output: {
    path: __dirname + '/dist',
    filename: 'index_bundle.js'
  },
  devServer: {
    contentBase: path.join(__dirname, 'dist')
  },
  module : {
    rules : [
      {
        test : /\.jsx$/,
        use : 'babel-loader'
      },
      {
        test : /\.css$/,
        use : ['style-loader', 'css-loader']
      },
      {
        test : /\.less$/,
        use : ['style-loader', 'css-loader', {
          loader : 'less-loader',
          options: {
            javascriptEnabled: true
          }
        }]
      }

    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      meta : {
        viewport : 'width=device-width'
      }
    })
  ]
}