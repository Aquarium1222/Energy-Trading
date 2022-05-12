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
    * 過去七天歷史用電資料
    * | Column Name |            Description             | Unit |
      |:-----------:| :--------------------------------: | :--: |
      |     time    |       Power Consumption Time       |      |
      | consumption | Household power consumption value  |  kWh |
  * generation.csv
    * 過去七天產電資料
    * | Column Name |            Description             | Unit |
      |:-----------:| :--------------------------------: | :--: |
      |     time    |       Power Consumption Time       |      |
      |  generation | Household power consumption value  |  kWh |
  * bidresult.csv
    * 過去七天自己的投標資料
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
    * 未來一天投標資訊
    * |  Column Name  |         Description          | Unit |
      |:-------------:|:----------------------------:|:----:|
      |     time      |   bidding time（目標競標時間）  | hour |
      |    action     |  bidding action（buy／sell）  |      |
      | target_price  |     bidding price（競標價）    | TWD  |
      | target_volume |     bidding volume（競標量）   | kWh  |

---

## 工作流程（Workflow）
<img src="./workflow.png" alt="Cover" width="50%"/>

---

