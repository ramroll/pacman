1. pacman.py 初始化的时候只更新pacmanType，和ghostType不具体初始化实例
2. classicGameRules.py.newGame的时候，会拿到pacmanType和ghostType
3. 在classicGameRules中实现一个方法根据GameState创建Agents，每个Agents都有它关联AgentState的引用 