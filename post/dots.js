const axios = require('axios')
const host = 'http://localhost:1080'
const params = {
  // 剩余的豆子的坐标 2
  pacDots: [
    {
      x: 3,
      y: 3
    },
    {
      x: 1,
      y: 1
    },
    {
      x: 1,
      y: 2
    },
    {
      x: 1,
      y: 3
    },
    {
      x: 1,
      y: 4
    },
    {
      x: 1,
      y: 5
    }
  ],
  // 剩余的大力丸的坐标 3
  powerPellets: [
    {
      x: 5,
      y: 1
    }
  ],
  // 剩余的pacman的坐标 4
  pacman: {
    0: {
      x: 3,
      y: 1
    }
  },
  // 剩余的ghost的坐标 5
  ghosts: {
    0: {
      x: 4,
      y: 5
    }
  },
  // 剩余的pacman的状态
  pacmanFeast: { 0: true }
}
axios
  // .get(host, { params })
  .post(host, params)
  .then((res) => {
    console.log('Res: ', res.data)
  })
  .catch((err) => {
    console.error(err)
  })
