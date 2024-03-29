# 結合 CNN 及深度影像於食物辨識及體積估算之中式自助餐計價系統
<img src="https://user-images.githubusercontent.com/22574508/228484288-eab6b0c4-0399-4efc-aebc-8b0c1687deaf.PNG" width="80%" height="80%"/>

## Introduction
![introduction](https://user-images.githubusercontent.com/22574508/228721720-47034007-5873-4981-9c52-00543f63c3bc.jpg)

## 實驗環境
- 架設一個燈管於頂端，用於確保光線一致性

<img src="https://user-images.githubusercontent.com/22574508/228476098-d65b4185-563a-4c00-a8f8-842d7e92be1c.jpg" width="50%" height="50%"/>
<img src="https://user-images.githubusercontent.com/22574508/228476117-8a0e3d70-0776-4837-9d2c-3fc96b31d3b4.jpg" width="50%" height="50%"/>



## 資料蒐集
- 學校學生餐廳蒐集自助餐食物
- 共22種菜色

- 每種菜色皆80~110張影像，共2025張
- 拍攝時皆經過攪動或翻面和不同食物量，確保資料的多樣性
- 使用Kinect v2鏡頭

<img src="https://user-images.githubusercontent.com/22574508/228472344-287f7e73-39e4-4c1e-a9f1-486becf676c7.PNG" width="50%" height="50%"/>

## 系統架構
- Server使用Flask框架開發
- Client使用WPF.NET開發
- Client使用Kinect v2鏡頭獲取彩色影像及深度資訊
- 使用MySQL建立資料庫

![System Architecture Diagram](https://user-images.githubusercontent.com/22574508/228471499-be7805c7-525f-4543-8ac4-caa303a00e4e.png)

## 詳細內容可詳閱本論文
連結：https://hdl.handle.net/11296/d38n5r

## Demo Video
[![Watch the video](https://img.youtube.com/vi/-JXHqa8L2zg/hqdefault.jpg)](https://youtu.be/-JXHqa8L2zg)
