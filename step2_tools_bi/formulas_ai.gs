/**
 * L2.1 Demo: Google Apps Script — AI generates spreadsheet formulas.
 *
 * Usage: Extensions → Apps Script → paste this code → Run askAI()
 * Requires: AI_PROVIDER_URL and AI_PROVIDER_TOKEN in Script Properties
 */

function askAI(prompt) {
  const props = PropertiesService.getScriptProperties();
  const apiUrl = props.getProperty("AI_PROVIDER_URL") || "https://api.agentplatform.ru/v1";
  const token = props.getProperty("AI_PROVIDER_TOKEN");

  const response = UrlFetchApp.fetch(apiUrl + "/chat/completions", {
    method: "post",
    contentType: "application/json",
    headers: { "Authorization": "Bearer " + token },
    payload: JSON.stringify({
      model: "openai/gpt-4o-mini",
      messages: [
        { role: "system", content: "Ты помощник для Google Sheets. Отвечай только формулой, без пояснений." },
        { role: "user", content: prompt }
      ],
      max_tokens: 500
    })
  });

  const data = JSON.parse(response.getContentText());
  return data.choices[0].message.content.trim();
}

/**
 * Demo: generate formula for average revenue by city
 */
function demoFormulaGeneration() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  // Ask AI to create a formula
  const prompt = "Google Sheets формула: средняя выручка (столбец J) для города Москва (столбец D), диапазон строк 2:501";
  const formula = askAI(prompt);

  // Write the formula to a cell
  sheet.getRange("M1").setValue("AI-формула:");
  sheet.getRange("N1").setFormula(formula);

  Logger.log("Generated formula: " + formula);
}

/**
 * Demo: conditional formatting via script
 */
function demoConditionalFormatting() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const range = sheet.getRange("J2:J501"); // revenue column

  // Green if above average, red if below
  const rules = sheet.getConditionalFormatRules();
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenNumberGreaterThan(5000)
      .setBackground("#d4edda")
      .setRanges([range])
      .build()
  );
  rules.push(
    SpreadsheetApp.newConditionalFormatRule()
      .whenNumberLessThan(1000)
      .setBackground("#f8d7da")
      .setRanges([range])
      .build()
  );
  sheet.setConditionalFormatRules(rules);
}
