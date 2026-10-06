# HK Flood Risk Map

香港積水風險指數互動地圖。

## 線上訪問

https://littlegut-gut.github.io/hk-flood-map/

## 技術架構

- 8 大風險因子（雨量、渠蓋、地勢、渠管、人口、沿海、土地、歷史黑點）
- 200m 網格 + 混合聚合
- AFCD 郊野公園邊界整合
- IDW 雨量插值
- 防洪設施風險下調
- 前端動態風險重算 + 動畫播放

## 自動更新

GitHub Actions 每小時從 HKO 抓取最新雨量，更新 `docs/data/rainfall_history.json`。
