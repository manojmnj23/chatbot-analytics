# RASA Chatbot Analytics

A comprehensive analytics project for RASA chatbot that provides insights into chatbot performance, intent recognition accuracy, user behavior, and conversion rates.

## Project Description

This project analyzes RASA chatbot logs and provides 4 different analytical outputs:
1. **Chatbot Dashboard** - Overall intent distribution and fallback rate analysis
2. **Intent Recognition Accuracy** - Evaluates how well the NLU model recognizes user intents
3. **User-Level Analysis** - Analyzes user behavior and fallback patterns
4. **Conversion & Satisfaction** - Calculates conversion rates and customer satisfaction metrics

## Project Structure

```
RASA_Chatbot_Analytics/
│
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore file
├── config.yml                          # RASA configuration
├── credentials.yml                     # RASA credentials
├── domain.yml                          # RASA domain definition
├── endpoints.yml                       # RASA endpoints configuration
│
├── Python Scripts (Analysis Tasks)
├── task1_chatbot_dashboard.py         # OUTPUT 1: Dashboard & fallback rate analysis
├── TASK_2_1.py                        # OUTPUT 2: Intent recognition accuracy evaluation
├── TASK_2_2.py                        # OUTPUT 3: User-level behavior analysis
├── Task_2_3.py                        # OUTPUT 4: Conversion & CSAT metrics
│
├── Data Files
├── rasa_logs.csv                      # RASA chatbot logs (input data)
├── intent_distribution.csv            # Generated: Overall intent distribution
├── user_intent_frequency.csv          # Generated: Intent frequency by user
│
├── RASA Training Data
├── data/
│   ├── nlu.yml                        # NLU training examples
│   ├── rules.yml                      # RASA rules
│   └── stories.yml                    # Conversation stories
│
├── Custom Actions
├── actions/
│   ├── __init__.py                    # Package initializer
│   └── actions.py                     # Custom RASA actions
│
├── Model & Cache
├── models/                            # Trained RASA models directory
├── .rasa/                             # RASA cache files
│
└── Tests
    └── tests/
        └── test_stories.yml           # RASA story tests
```

## Outputs

### OUTPUT 1: task1_chatbot_dashboard.py
**Purpose**: Chatbot Dashboard & Fallback Rate Analysis

**Key Metrics**:
- Fallback Rate: 14.29%
- Total User Queries: 49
- Top Bot Actions: action_listen (52), action_unlikely_intent (22), utter_goodbye (22)

**What it shows**:
- User intent distribution visualization
- Bot action frequency analysis
- Fallback queries by user
- User-Intent frequency table

### OUTPUT 2: TASK_2_1.py
**Purpose**: Intent Recognition Accuracy Evaluation

**Key Metrics**:
- Intent Recognition Accuracy: 36.73%
- Best performing intents: greet (91% F1-score), shipping_info (100% F1-score)
- Worst performing intents: deny, mood_unhappy, nlu_fallback (0% F1-score)

**What it shows**:
- Comparison of RASA predicted intents vs true intents
- Classification report with precision, recall, and F1-scores
- Per-intent performance breakdown

### OUTPUT 3: TASK_2_2.py
**Purpose**: User-Level Behavior & Fallback Analysis

**Key Metrics**:
- Number of Unique Users: 3
- Total User Queries: 49
- Number of Unique Intents: 7
- Overall Fallback Rate: 61.22%
- Highest fallback user: User 5625fb0f7b3b425c82867f2d7b490ff6 (73.68%)

**What it shows**:
- User-level fallback rates and patterns
- Intent frequency distribution
- Session length statistics
- Intent comparison (RASA vs True intents)

### OUTPUT 4: Task_2_3.py
**Purpose**: Conversion Rate & Customer Satisfaction Metrics

**Key Metrics**:
- Overall Conversion Rate: 100%
- Estimated CSAT (Customer Satisfaction): 100%
- NPS Score (Net Promoter Score): 100

**What it shows**:
- Per-user conversion rates
- Customer satisfaction estimation
- NPS score calculation

## Requirements

- Python 3.8+
- RASA 3.0+
- pandas
- matplotlib
- seaborn
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/manojmnj23/chatbot-analytics.git
cd RASA_Chatbot_Analytics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn rasa
```

## Running the Analysis

Run all 4 analysis scripts:

```bash
# OUTPUT 1: Chatbot Dashboard
python task1_chatbot_dashboard.py

# OUTPUT 2: Intent Recognition Accuracy
python TASK_2_1.py

# OUTPUT 3: User-Level Analysis
python TASK_2_2.py

# OUTPUT 4: Conversion & CSAT
python Task_2_3.py
```

Or run all at once:
```bash
python task1_chatbot_dashboard.py && python TASK_2_1.py && python TASK_2_2.py && python Task_2_3.py
```

## Key Insights

1. **Fallback Rate**: The chatbot has a 14-61% fallback rate depending on the metric used, indicating room for NLU model improvement.

2. **Intent Recognition**: Low accuracy (36.73%) suggests the NLU model needs more training data or better feature engineering.

3. **User Patterns**: One user (ID: 5625fb0f7b3b425c82867f2d7b490ff6) has significantly higher fallback rate (73.68%), indicating potential issues with specific user behavior patterns.

4. **High Satisfaction**: Despite lower accuracy metrics, conversion and CSAT scores are high (100%), suggesting user tolerance or data collection issues.

## Files Description

- **rasa_logs.csv**: Main input data file containing all chatbot interactions with timestamps, sender IDs, intents, and actions
- **config.yml**: RASA pipeline and component configuration
- **domain.yml**: Defines intents, entities, actions, and responses
- **data/**: Contains NLU training data (examples, rules, and stories)
- **actions/**: Custom action implementations for the chatbot

## Future Improvements

- Collect more training data for NLU model
- Implement advanced NLU components for better intent recognition
- Add entity extraction analysis
- Create real-time dashboards with Streamlit/Dash
- Implement A/B testing framework for chatbot improvements

## License

This project is open source and available under the MIT License.

## Author

- **manoj**

## Support

For issues or questions, please open an issue on the GitHub repository.
