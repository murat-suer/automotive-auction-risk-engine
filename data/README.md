# Data

`data_train.csv` (~12 MB, tracked with **Git LFS**) — one row per used vehicle
that was bought at auction and later resold. The target is `IsBadBuy`
(1 = lemon / "Montagsauto", 0 = good buy). Classes are imbalanced (~12 % lemons).

`PurchDate` is stored as epoch seconds in the file and parsed to datetime by
[`auto_risk.data.load_data`](../src/auto_risk/data.py).

## Source & attribution

The dataset is derived from the public Carvana *"Don't Get Kicked!"* Kaggle
competition on used-car auction lemon prediction. It is included here so the
pipeline is reproducible end to end. All rights to the underlying data remain
with the original publishers; it is redistributed for educational/portfolio use.

## Data dictionary (33 columns)

| Column | Type | Description |
|---|---|---|
| `IsBadBuy` | categorical | Target — 1 = lemon (bad buy), 0 = good buy |
| `PurchDate` | datetime | Auction purchase date (epoch seconds in the file) |
| `Auction` | categorical | Auction provider |
| `VehYear` | int | Vehicle model year |
| `VehicleAge` | int | Age at auction (years) |
| `Make` | categorical | Manufacturer |
| `Model` | categorical | Model |
| `Trim` | categorical | Trim level |
| `SubModel` | categorical | Sub-model |
| `Color` | categorical | Colour |
| `Transmission` | categorical | Automatic / Manual |
| `WheelTypeID` | categorical | Wheel-type id |
| `WheelType` | categorical | Wheel type |
| `VehOdo` | int | Odometer (miles) |
| `Nationality` | categorical | Manufacturer's country |
| `Size` | categorical | Size class (compact, SUV, …) |
| `TopThreeAmericanName` | categorical | One of the top-3 US makers? |
| `MMRAcquisitionAuctionAveragePrice` | int | Acquisition auction price, average condition |
| `MMRAcquisitionAuctionCleanPrice` | int | Acquisition auction price, clean condition |
| `MMRAcquisitionRetailAveragePrice` | int | Acquisition retail price, average condition |
| `MMRAcquisitonRetailCleanPrice` | int | Acquisition retail price, clean condition |
| `MMRCurrentAuctionAveragePrice` | int | Current auction price, average condition |
| `MMRCurrentAuctionCleanPrice` | int | Current auction price, clean condition |
| `MMRCurrentRetailAveragePrice` | int | Current retail price, average condition |
| `MMRCurrentRetailCleanPrice` | int | Current retail price, clean condition |
| `PRIMEUNIT` | categorical | Higher-than-standard demand? |
| `AUCGUART` | categorical | Auction guarantee level (GREEN/YELLOW/RED) |
| `BYRNO` | categorical | Unique buyer number |
| `VNZIP1` | categorical | Purchase ZIP code |
| `VNST` | categorical | Purchase state |
| `VehBCost` | int | Acquisition cost paid (USD) |
| `IsOnlineSale` | categorical | Originally bought online? |
| `WarrantyCost` | int | 36-month warranty cost (USD) |

The `MMR*` columns are eight correlated market-reference prices; the pipeline
compresses them with PCA(2) to curb multicollinearity (see
[`auto_risk.preprocessing`](../src/auto_risk/preprocessing.py)).
