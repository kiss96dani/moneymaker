# ML Pipeline Documentation

## Overview

This document describes the machine learning pipeline integrated into the moneymaker betting analysis system. The pipeline extends the existing Poisson-based predictions with trained LogisticRegression models for more accurate market predictions.

## Architecture

### Components

1. **models/trainer.py**: Handles historical data fetching, feature engineering, and model training
2. **models/model_wrapper.py**: Provides model loading and prediction interface
3. **predictor.py**: Extended to support both Poisson and ML-based predictions with automatic fallback
4. **apifootball_client.py**: Extended with date-range fixture fetching for training data

### Prediction Flow

```
User Input
    ↓
Predictor (ML mode enabled?)
    ↓
    ├─→ ML Models Available?
    │   ├─→ Yes: Use ML predictions
    │   └─→ No: Fall back to Poisson
    ↓
Market Odds Parsing
    ↓
Edge & Kelly Calculations
    ↓
Results
```

## Features Engineered

The ML models use the following features:

| Feature | Description | Source |
|---------|-------------|--------|
| `home_goals_per_match` | Average goals scored by home team (last 5 matches) | Historical fixtures |
| `home_goals_against_per_match` | Average goals conceded by home team | Historical fixtures |
| `away_goals_per_match` | Average goals scored by away team (last 5 matches) | Historical fixtures |
| `away_goals_against_per_match` | Average goals conceded by away team | Historical fixtures |
| `form_score_home` | Weighted form score (W=1, D=0.5, L=0) | Recent results |
| `form_score_away` | Weighted form score for away team | Recent results |
| `home_adv` | Home advantage constant | Environment variable |

**Note**: Recent matches are weighted more heavily (0.6, 0.7, 0.8, 0.9, 1.0 from oldest to newest).

## Models

### 1X2 Model
- **Type**: Multinomial Logistic Regression
- **Classes**: 3 (home win, draw, away win)
- **Output**: Probabilities for each outcome

### BTTS Model
- **Type**: Binary Logistic Regression  
- **Classes**: 2 (both teams score: yes/no)
- **Output**: Probability of BTTS

### Over/Under 2.5 Model
- **Type**: Binary Logistic Regression
- **Classes**: 2 (over/under 2.5 goals)
- **Output**: Probability of over 2.5 goals

All models use:
- StandardScaler for feature normalization
- lbfgs solver
- max_iter=1000
- random_state=42 for reproducibility

## Training Workflow

### Step 1: Prepare Training Data

Choose your parameters:
- **Date range**: Historical period to train on (recommend 6-12 months)
- **Leagues**: List of league IDs (e.g., 39=Premier League, 61=Ligue 1)

### Step 2: Run Training

```bash
python betting.py --train-models \
  --train-from 2023-01-01 \
  --train-to 2023-12-31 \
  --leagues 39,61
```

This will:
1. Fetch all completed fixtures for specified leagues and date range
2. For each fixture, fetch team form data before that match
3. Engineer features from historical performance
4. Train three models (1X2, BTTS, Over/Under 2.5)
5. Save models to `models/` directory:
   - `feature_scaler.pkl`
   - `1x2.pkl`
   - `btts.pkl`
   - `over25.pkl`

**Important**: Training generates many API calls. Start with a shorter period and fewer leagues.

### Step 3: Verify Training

Check that model files exist:
```bash
ls -la models/*.pkl
```

You should see:
```
models/1x2.pkl
models/btts.pkl
models/feature_scaler.pkl
models/over25.pkl
```

## Prediction Workflow

### Using ML Models

Once models are trained, enable ML mode:

```bash
# Fetch and analyze with ML
python betting.py --fetch --analyze --use-ml

# Analyze specific fixtures with ML
python betting.py --analyze --fixture-ids 12345,67890 --use-ml
```

### Fallback Behavior

If models are not available or fail to load:
1. System logs a warning
2. Automatically falls back to Poisson predictions
3. Analysis continues without interruption

This ensures the system is always operational.

## Odds Parsing

The improved odds parser handles multiple market types:

### Supported Markets

| Market | API Label Variants | Output |
|--------|-------------------|---------|
| 1X2 | "1X2", "Match Winner", "Match Odds" | home_odds, draw_odds, away_odds |
| BTTS | "BTTS", "Both Teams", "Both Teams Score" | btts_yes_odds, btts_no_odds |
| Over/Under 2.5 | "Over/Under", "Goals Over/Under" | over25_odds, under25_odds |

### Output Format

```python
{
    "market_odds": {
        "home_odds": 2.10,
        "draw_odds": 3.40,
        "away_odds": 3.50,
        "btts_yes_odds": 1.80,
        "btts_no_odds": 2.00,
        "over25_odds": 1.90,
        "under25_odds": 1.95
    },
    "market_probs": {
        "home": 0.4762,
        "draw": 0.2941,
        "away": 0.2857,
        "btts_yes": 0.5556,
        "btts_no": 0.5000,
        "over25": 0.5263,
        "under25": 0.5128
    }
}
```

## Edge Calculation

For each market where both model probability and market odds are available:

### Formula

```
Edge = (Model_Probability × Odds) - 1
Kelly_Fraction = Edge / (Odds - 1)  [if Edge > 0]
Recommended_Stake = Kelly_Fraction × Bankroll
```

### Example

```
Model predicts: Home win 70%
Market odds: 2.00 (implies 50%)

Edge = (0.70 × 2.00) - 1 = 0.40 (40% edge!)
Kelly = 0.40 / (2.00 - 1) = 0.40 (40% of bankroll)
Stake = 0.40 × $1000 = $400
```

**Note**: Kelly is capped at 100% (full bankroll) for safety. Negative edges result in 0 stake recommendation.

## Configuration

Set these environment variables in `.env`:

```bash
# API key (required)
API_FOOTBALL_KEY=your_key_here

# Home advantage factor (default: 0.20)
HOME_ADV=0.20

# Daily bankroll for Kelly calculations (default: 1000)
BANKROLL_DAILY=1000

# Days ahead to fetch (default: 3)
FETCH_DAYS_AHEAD=3
```

## API Rate Limits

Training is API-intensive. For each fixture in training data:
- 1 API call to fetch the fixture
- 2 API calls to fetch team form (home and away)

Example: Training on 500 fixtures = ~1000 API calls

**Recommendations**:
- Start with 3-6 months of data
- Use 1-2 leagues initially
- Consider API plan limits
- Future enhancement: Add retry logic with exponential backoff

## Model Performance

After training, the system outputs training accuracy:

```
[INFO] 1X2 training accuracy: 0.534
[INFO] BTTS training accuracy: 0.612
[INFO] Over/Under 2.5 training accuracy: 0.587
```

**Note**: These are in-sample training accuracies. For production use, implement proper train/test splits and cross-validation.

## Troubleshooting

### Models Won't Load

**Symptom**: `[WARNING] Not all model files found. ML mode unavailable.`

**Solution**: 
1. Check `models/` directory exists
2. Verify all 4 .pkl files are present
3. Re-run training if files are missing

### Low Training Accuracy

**Symptom**: Accuracy < 40%

**Possible causes**:
- Insufficient training data
- Poor quality historical data
- Feature engineering needs improvement

**Solutions**:
- Increase training date range
- Add more leagues
- Verify API data quality

### API Rate Limit Errors

**Symptom**: Training fails with HTTP 429 errors

**Solution**:
- Reduce date range
- Reduce number of leagues
- Add delays between requests (already has 0.1s throttle)
- Upgrade API plan

## Future Enhancements

Potential improvements to the ML pipeline:

1. **Model improvements**:
   - Cross-validation
   - Hyperparameter tuning
   - Ensemble methods
   - Neural networks

2. **Feature engineering**:
   - Head-to-head history
   - Player availability
   - Recent injuries
   - League position/strength

3. **Robustness**:
   - Retry logic with exponential backoff
   - Caching of training data
   - Incremental training

4. **Evaluation**:
   - Hold-out test set
   - ROI tracking
   - Model calibration

## License & Disclaimer

This software is for educational purposes. Sports betting involves financial risk. Always gamble responsibly and within your means.
