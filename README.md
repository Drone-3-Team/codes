# codes

lidar.py是控制飞机飞行、获取传感器数据的python文件

setting.json替换用户目录/Airsim下的jason文件，作用是定义飞机的构型，给它加上一个激光雷达

## 添加TopDownInteriors的方法

  1、下载解压
  2、新建一个导入了airsim的项目或者直接用之前弄好的LandscapeMountains（推荐后者）
  3、打开项目文件夹找到Content，把TopDownInteriors文件夹直接放进去
  4、用VS打开项目运行进入虚幻编辑器，下方就可以找到TopDownInteriors的文件夹
  5、打开里面的场景，并把游戏模式重定义成Airsim Gamemode

~属于是回忆中的大致方法，有一些名称可能不甚准确，但大意没错~
