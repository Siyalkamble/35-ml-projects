<h2>Project 7: Simple API Consumer </h2>
Tech: Requests, JSON, APIs <br>
What You Build:<br> A program using requests library that calls a free public API (OpenWeather, PokéAPI, etc.), parses JSON response, and displays results beautifully.<br>
Exact Steps:<br> 1. Choose free public API (OpenWeatherMap, REST Countries, PokéAPI) <br>
2. Get API key (if required) <br>
3. Make GET request to API<br>
4. Parse JSON response <br>
5. Extract relevant data (temperature, currency rate,Pokémon stats) <br>
6. Display in nice format (formatted table or text) <br>
7. Handle errors gracefully (connection issues, invalid requests) <br>
8. Cache results to avoid excessive API calls <br>
Learning Goals : <br>
- HTTP requests (GET, POST) <br>
- Working with APIs - JSON parsing <br>
- Error handling - Rate limiting concepts<br>
Real-World Use: <br>Weather apps, stock price apps, currency converter, any app that uses external data

<h3>Problems that I faced</h3>
1. I thought I can just use 
` res.text ` 
and it will automatically save the file in dict/json but I was wrong it returns str 
2. Also I make a typo in ` data = json.load ` instead of writting ` json.loads `
3. In the main part I couldn't figure out how to get the Pokemon types which was in the nested dictionary I was doing manually like first geting the first type and then check if second one exists or not but here I messed up with the syntax and couldn't figure out how to do this part then I asked for the hint in AI not the actual code 