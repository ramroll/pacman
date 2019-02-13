const axios = require('axios')
const host = 'http://localhost:1080'
const params = {
  config: {
    round: 2000, // 本局比赛的回合限制
    feastTime: 5, // pacman超级状态可以维持的回合数
    lives: 3,
    pacDotPoint: 1, // 一颗豆子的分值
    powerPelletPoint: 1, // 一颗大力丸的分值
    pacmanPoint: 5, // 一个pacman的分值
    ghost: 5 // 一个ghost的分值
  },
  map: {
    height: 7,
    width: 7,
    pixels: [
      [1, 1, 1, 1, 1, 1, 1],
      [1, 0, 4, 0, 0, 0, 1],
      [1, 0, 1, 1, 1, 0, 1],
      [1, 0, 1, 2, 0, 0, 1],
      [1, 0, 1, 1, 1, 0, 1],
      [1, 0, 0, 5, 0, 0, 1],
      [1, 1, 1, 1, 1, 1, 1]
    ]
  }
}
axios
  // .get(host, { params })
  .post(host, params)
  .then((res) => {
    // console.log('Res: ', res)
  })
  .catch((err) => {
    console.error(err)
  })
