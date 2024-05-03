## ETF Analyzer
Autogen bot that analyzes how attractive a given ETF is given the current market and macroeconomic conditions

Work in progress.

### To run
Create a file called api_keys.json that looks like this:

```
{
  "FRED": "FRED_API_KEY (https://fred.stlouisfed.org/docs/api/fred/)",
  "OPENAI": "OPENAI_API_KEY" (https://openai.com/index/openai-api),
  "GOOGLE": "GOOGLE_API_KEY" (https://console.cloud.google.com/apis/credentials),
  "SEI": "SEI_API_KEY (https://programmablesearchengine.google.com/controlpanel/create)"
}
```

python ETF_evaluator.py VBR --risk high --save_file VBR_chat_log.txt
