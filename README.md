# Energy-Trading

#### DSAI_HW3-Energy-Trading

[Guideline](https://docs.google.com/presentation/d/1ZwXe4xMflCxiDQ7RK6z_LH88r0Dp38sQ/edit#slide=id.p1)

[交易資訊線上表單](https://docs.google.com/spreadsheets/d/1hqoxG48A159buQ-GuoU7Fo-QrGKYmE1DFgPckJR0dFI/edit?usp=sharing)

---

test the code by running 

```python main.py --consumption "Consumption Data" --generation "Generation Data" --bidresult "Bidresult Data" --output output.csv```

---

## 概述（Overview）

* 我所扮演的角色是一個房屋所有者，而我的房子具有電器（例如電燈、冷氣......等等），也具備太陽能發電器。

* 我的房子同時也在一個具有 40 戶的社區微電網中，我可以決定透過電力交易平台購買或出售微電網的電力，也可以依據固定價格購買主電網的電力。

* 目標：設計一個 agent 來進行競標電力，希望最終能使電費最小化。

---

## 資料（Data）

1. 50 households synthetic data
2. Hourly Power consumption (kWh)
3. Hourly Solar Power generation (kWh)
4. Hourly Bidding records
  * 12 months of data
    * 8 months for training [training_data]()
      * target0.csv ~ target49.csv 共 50 筆 csv 檔，每份 header 皆為 time, generation, consumption
    * 1 months for validation
    * 3 months for testing

---

## 輸入與輸出（Input and Output）

* input  
  * consumption.csv
    * 用戶之過去七天的用電資料（每小時一筆），共 168（ 7 * 24 = 168 ）筆
    * | Column Name |            Description             | Unit |
      |:-----------:| :--------------------------------: | :--: |
      |     time    |       Power Consumption Time       |      |
      | consumption | Household power consumption value  |  kWh |
  * generation.csv
    * 用戶之過去七天的產電資料（每小時一筆），共 168（ 7 * 24 = 168 ）筆
    * | Column Name |            Description             | Unit |
      |:-----------:| :--------------------------------: | :--: |
      |     time    |       Power Consumption Time       |      |
      |  generation | Household power consumption value  |  kWh |
  * bidresult.csv
    * 過去七天自己的投標資料
    * 包含詳細的交易結果資訊，第一天的資訊為空，第二天開始會加入前一天的投標資料（如 [workflow.png](https://github.com/Aquarium1222/Energy-Trading/blob/main/workflow.png) 所示）
    * |  Column Name  |                  Description                  | Unit |
      |:-------------:|:---------------------------------------------:|:----:|
      |     time      |           bidding time（目標競標時間）           | hour |
      |    action     |           bidding action (buy／sell)           |      |
      | target_price  |             bidding price (投標價)              | TWD  |
      | target_volume |             bidding volume (投標量)             | kWh  |
      |  trade_price  |  closing price (成交價), -1 when bidding fails  | TWD  |
      | trade_volume  |             closing volume (成交量)             | kWh  |
      |    status     |    bidding result (未成交／完全成交／部分成交）     |      |

* output
  * output.csv
    * 輸出未來一天的時間、決策（買、賣、無競價行動）、目標價格、目標量
    * 同時段可以買賣，但每日最多投標 100 筆
    * |  Column Name  |         Description          | Unit |
      |:-------------:|:----------------------------:|:----:|
      |     time      |   bidding time（目標競標時間）  | hour |
      |    action     |  bidding action（buy／sell）  |      |
      | target_price  |     bidding price（競標價）    | TWD  |
      | target_volume |     bidding volume（競標量）   | kWh  |

---

## 工作流程（Workflow）
<img src="./workflow.png" alt="Cover" width="50%"/>

* 平台會指定過去七天的資料（cosumption, generation, bidresult）作為參數放進 Agents 程式中
* Agents 會產出未來一天的投標資訊（以小時為單位，意即一天會有 24 筆資料），並可以透過參數指定輸出路徑
* 平台會取得所有 Agents 的投標資訊，並進行媒合
* 平台公告結果並將競標結果寫到各個 Agents 的目錄

---

## 競價行動的類型（Action Type）

* 買
  * 找到一個賣家來購買你的電力，並以目標數量（target_volume）和目標價格（target_price）購買
  * 如果交易成功（成交），買方會獲得對應的交易量（trade_volume）和交易價格

* 賣
  * 以目標數量（target_volume）和目標價格（target_price）尋找買家來購買我所想要賣掉的電力
  * 如果賣方賣給買方的電量不足，賣方就需要向台電購買電量，並以市電單價（market_price）供應給買方
    * 此處的 market_price 為 2.5256

* 無競價行動
  * 在這些時間中，不向 output.csv 輸出任何競價行動

---

## 電費計算（Electricity Bill）

* 總電費計算方式為 A + B
  * A: 從平台 賺／花 的錢
    * a = 買電量 - 賣電量 + 產電量
      * a >= 0: A =（買 - 賣）* 成交價
      * a < 0: A =（-1）* （買 - 賣）* 成交價
  * B: 正常使用電費的錢
    * b = a - 用電量
      * b >= 0: B = 0
      * b < 0: B =（-1）* b * 市電單價

---

## 競價規則（Bidding Rules）

* 同時段可以買賣，每日最多投標 100 筆
* 以小時為單位媒合，格式：%Y-%m-%d %H:%M:%S
* 投標價格與投標量皆以 0.01 為單位（四捨五入）

---

## 分析（Analysis）

* 會影響電費計算之主要因素
  * 自身的電力
    * 用電量
    * 產電量
  * 競價行動
    * 買賣量（volume）
    * 買賣價格（price）

---

## 用電量與產電量（consumption and generation）




---

