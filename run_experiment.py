import openai
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import json
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

# Set your API key
# client = openai.OpenAI(api_key="your-api-key-here")  # Uncomment and add your key
client = openai.OpenAI()  # This will use the OPENAI_API_KEY environment variable

TEMPERATURE = 0.0

def get_confidence_and_logprobs(question, model="gpt-4o"):
    """
    Get yes/no answer with self-evaluated confidence AND token logprobs.
    
    Args:
        question (str): The question to ask the model
        model (str): The OpenAI model to use
    
    Returns:
        dict: Contains the answer, self-evaluated confidence, and logprobs
    """
    # Clear instruction for yes/no with confidence rating
    system_message = """You are a helpful assistant that responds to yes/no questions.
    
    IMPORTANT: Your response must ALWAYS follow this exact format:
    "Yes. Confidence: X%" or "No. Confidence: X%"
    
    - Your answer must start with either "Yes" or "No" as the very first word.
    - After your Yes/No, include a period, then "Confidence: " followed by a percentage.
    - The percentage (X%) should reflect how confident you are in your answer.
    - Do not include any other text or explanation in your response.
    
    Example response: "Yes. Confidence: 85%"
    """
    
    # First API call to get the model's self-evaluated confidence
    response_full = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        temperature=TEMPERATURE
    )
    
    full_response = response_full.choices[0].message.content.strip()
    
    # Parse the model's answer and confidence
    answer_pattern = r'^(Yes|No)\. Confidence: (\d+)%$'
    match = re.match(answer_pattern, full_response)
    
    if not match:
        print(f"Warning: Response format incorrect: '{full_response}'")
        # Try to extract as best we can
        if full_response.lower().startswith("yes"):
            self_answer = "Yes"
        elif full_response.lower().startswith("no"):
            self_answer = "No"
        else:
            self_answer = "Unknown"
            
        confidence_match = re.search(r'Confidence: (\d+)%', full_response)
        if confidence_match:
            self_confidence = int(confidence_match.group(1))
        else:
            self_confidence = None
    else:
        self_answer = match.group(1)
        self_confidence = int(match.group(2))
    
    # Second API call to get token logprobs for the first token
    response_logprobs = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        temperature=0,  # Use temperature=0 for deterministic response
        logprobs=True,
        top_logprobs=5,  # Return logprobs for top 5 tokens
        max_tokens=1     # We only need one token - Yes or No
    )
    
    # Get content and log probabilities
    content = response_logprobs.choices[0].message.content.strip().lower()
    logprobs = response_logprobs.choices[0].logprobs
    
    # Get top tokens and their logprobs from first position
    token_logprobs = logprobs.content[0].top_logprobs
    
    # Create a dictionary of token -> logprob
    token_logprob_dict = {item.token.lower(): item.logprob for item in token_logprobs}
    
    # Extract yes/no logprobs (handling different variations)
    yes_tokens = ['yes', ' yes', 'y', ' y', 'yes.']
    no_tokens = ['no', ' no', 'n', ' n', 'no.']
    
    # Find the best yes and no token logprobs
    yes_logprob = max([token_logprob_dict.get(t, -100) for t in yes_tokens], default=-100)
    no_logprob = max([token_logprob_dict.get(t, -100) for t in no_tokens], default=-100)
    
    # Get top logprob overall (might be neither yes nor no)
    top_token, top_logprob = max(token_logprob_dict.items(), key=lambda x: x[1])
    is_other = (top_token.lower() not in yes_tokens and top_token.lower() not in no_tokens)
    
    # Convert to probabilities
    yes_prob = math.exp(yes_logprob)
    no_prob = math.exp(no_logprob)
    
    # Calculate total probability mass for yes, no, and other tokens
    other_tokens = {t: lp for t, lp in token_logprob_dict.items() 
                   if t.lower() not in yes_tokens and t.lower() not in no_tokens}
    other_probs = sum([math.exp(lp) for lp in other_tokens.values()])
    
    # Normalize to get proper probabilities that sum to 1
    total = yes_prob + no_prob + other_probs
    yes_prob_normalized = yes_prob / total
    no_prob_normalized = no_prob / total
    other_prob_normalized = other_probs / total
    
    # Calculate logprob confidence (difference between yes and no)
    logp_diff = abs(yes_logprob - no_logprob)
    
    # Determine which probability to use based on the answer
    if self_answer == "Yes":
        answer_logprob = yes_logprob
        logp_confidence = yes_prob_normalized * 100
    elif self_answer == "No":
        answer_logprob = no_logprob
        logp_confidence = no_prob_normalized * 100
    else:
        answer_logprob = None
        logp_confidence = None
    
    return {
        "question": question,
        "full_response": full_response,
        "self_answer": self_answer,
        "self_confidence": self_confidence,
        "logprob_answer": "Yes" if yes_prob > no_prob else "No",
        "logp_confidence": logp_confidence,
        "yes_logprob": yes_logprob,
        "no_logprob": no_logprob,
        "yes_probability": yes_prob_normalized * 100,
        "no_probability": no_prob_normalized * 100,
        "other_probability": other_prob_normalized * 100,
        "content": content,
        "top_tokens": token_logprob_dict,
        "logp_diff": logp_diff
    }

def create_bar_chart_comparison(results, output_file='confidence_comparison_bar.png'):
    """Create an improved bar chart comparing self-confidence vs logp-confidence"""
    # Data preparation
    self_confidence = [r["self_confidence"] for r in results]
    logp_confidence = [r["logp_confidence"] for r in results]
    
    # Setup the figure
    plt.figure(figsize=(14, 10))
    
    # Create the bar chart
    ind = np.arange(len(results))
    width = 0.35
    
    plt.bar(ind - width/2, self_confidence, width, label='Self-evaluated', color='#4285F4', alpha=0.8)
    plt.bar(ind + width/2, logp_confidence, width, label='LogP-derived', color='#EA4335', alpha=0.8)
    
    plt.title('Self-evaluated vs LogP-derived Confidence', fontsize=16)
    plt.ylabel('Confidence (%)', fontsize=12)
    plt.xlabel('Question Number', fontsize=12)
    plt.xticks(ind, range(1, len(results) + 1))
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # Print the questions with their numbers
    print("Questions:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['question']}")

def create_scatter_correlation(results, output_file='confidence_correlation_scatter.png'):
    """Create an improved scatter plot with regression line to check correlation"""
    # Data preparation
    self_confidence = [r["self_confidence"] for r in results]
    logp_confidence = [r["logp_confidence"] for r in results]
    
    # Setup the figure
    plt.figure(figsize=(10, 8))
    
    # Create the scatter plot with numbered points
    plt.scatter(self_confidence, logp_confidence, s=80, alpha=0.7, c='#4285F4', edgecolors='black')
    
    # Add numbers to each point
    for i, (x, y) in enumerate(zip(self_confidence, logp_confidence)):
        plt.annotate(str(i+1), (x, y), fontsize=9, 
                    ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', alpha=0.8))
    
    # Add regression line
    z = np.polyfit(self_confidence, logp_confidence, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(self_confidence)-5, max(self_confidence)+5, 100)
    plt.plot(x_range, p(x_range), "--", color='#EA4335', linewidth=2, alpha=0.8)
    
    # Calculate correlation coefficient
    corr = np.corrcoef(self_confidence, logp_confidence)[0, 1]
    
    # Add diagonal line y=x
    min_val = min(min(self_confidence), min(logp_confidence)) - 5
    max_val = max(max(self_confidence), max(logp_confidence)) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.5, linewidth=1.5)
    
    # Set better axis limits to zoom in on the data
    buffer = 5  # Add some buffer around the data points
    x_min = max(0, min(self_confidence) - buffer)
    x_max = min(100, max(self_confidence) + buffer)
    y_min = max(0, min(logp_confidence) - buffer)
    y_max = min(100, max(logp_confidence) + buffer)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Labels and styling
    plt.title(f'Correlation between Self-evaluated and LogP-derived Confidence\nr = {corr:.3f}', fontsize=14)
    plt.xlabel('Self-evaluated Confidence (%)', fontsize=12)
    plt.ylabel('LogP-derived Confidence (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a note about what the numbers represent
    note = "Note: Numbers represent question indices"
    plt.figtext(0.5, 0.01, note, wrap=True, horizontalalignment='center', fontsize=10)
    
    # Use integer ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Tight layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # Print the questions with their numbers
    print("\nQuestions:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['question']}")

def create_comparison_table(results):
    """Create a markdown table with comparison data"""
    table = "| # | Question | Self Answer | Self Conf | LogP Answer | LogP Conf | Diff | LogP Diff |\n"
    table += "|---|---------|-------------|-----------|-------------|-----------|------|----------|\n"
    
    for i, r in enumerate(results):
        question = r["question"][:40] + "..." if len(r["question"]) > 40 else r["question"]
        diff = r["self_confidence"] - r["logp_confidence"]
        logp_diff = r["logp_diff"]
        
        # Highlight disagreements in answers
        answer_match = "✓" if r["self_answer"] == r["logprob_answer"] else "✗"
        
        table += f"| {i+1} | {question} | {r['self_answer']} {answer_match} | {r['self_confidence']}% | {r['logprob_answer']} | {r['logp_confidence']:.1f}% | {diff:.1f}% | {logp_diff:.2f} |\n"
    
    return table

def analyze_results(results):
    """Analyze the results to detect patterns and correlations"""
    # Convert to numpy arrays for easier analysis
    self_confidence = np.array([r["self_confidence"] for r in results])
    logp_confidence = np.array([r["logp_confidence"] for r in results])
    logp_diffs = np.array([r["logp_diff"] for r in results])
    
    # Calculate correlations
    corr_conf = np.corrcoef(self_confidence, logp_confidence)[0, 1]
    corr_diff = np.corrcoef(logp_diffs, self_confidence)[0, 1]
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(self_confidence - logp_confidence))
    
    # Calculate percentage of matching answers
    answer_matches = sum(1 for r in results if r["self_answer"] == r["logprob_answer"])
    match_percentage = (answer_matches / len(results)) * 100
    
    # Find questions with largest discrepancies
    discrepancies = [(i, abs(r["self_confidence"] - r["logp_confidence"])) 
                    for i, r in enumerate(results)]
    largest_discrepancies = sorted(discrepancies, key=lambda x: x[1], reverse=True)[:3]
    
    analysis = {
        "correlation_confidence": corr_conf,
        "correlation_logp_diff": corr_diff,
        "mean_absolute_error": mae,
        "answer_match_percentage": match_percentage,
        "largest_discrepancies": [results[i]["question"] for i, _ in largest_discrepancies],
        "avg_self_confidence": np.mean(self_confidence),
        "avg_logp_confidence": np.mean(logp_confidence)
    }
    
    return analysis

# Example usage
if __name__ == "__main__":
    # A range of questions with varying degrees of certainty
    questions = [
        "Is pineapple an appropriate pizza topping?",
        "Is it morally acceptable to lie to protect someone's feelings?",
        "Are cats better pets than dogs?",
        "Is modern art valuable?",
        "Should society prioritize individual freedom over collective welfare?",
        "Does free will truly exist?",
        "Is it ethical to eat meat?",
        "Are superhero movies good cinema?",
        "Is mathematics discovered or invented?",
        "Should schools eliminate homework completely?",
        "Is it better to be kind or to be right?",
        "Does true altruism exist?",
        "Is technology making people more or less connected?",
        "Is a minimalist lifestyle superior to a materialistic one?",
        "Should tipping be eliminated in restaurants?",
        "Is breakfast the most important meal of the day?",
        "Is it better to live in a city than in a rural area?",
        "Should people generally trust their intuition over data?",
        "Is privacy more important than convenience?",
        "Are social media platforms overall beneficial to society?",
        "Is nature more influential than nurture in determining personality?",
        "Should all education be free?",
        "Is it unethical to eat animal products?",
        "Is love primarily a chemical reaction?",
        "Should voting be mandatory?",
        "Is capitalism the best economic system?",
        "Are aliens likely to exist?",
        "Is professional sports a positive influence on culture?",
        "Should all drugs be decriminalized?",
        "Is it wrong to genetically modify human embryos?",
        "Should artificial intelligence development be restricted?",
        "Is democracy always the best form of government?",
        "Is there objective truth in art criticism?",
        "Should parents monitor their teenagers' online activities?",
        "Is it ethical to use animals in scientific research?",
        "Should healthcare be completely free?",
        "Is it better to specialize or be a generalist in your career?",
        "Can money buy happiness?",
        "Is nuclear energy a good solution for climate change?",
        "Is there such a thing as a just war?",
        "Should college athletes be paid?",
        "Is social media making people more narcissistic?",
        "Should religious institutions pay taxes?",
        "Is home ownership better than renting?",
        "Is it moral to eat meat if you could choose not to?"
    ]
    
    results = []
    print(f"Running experiment with {len(questions)} questions...")
    
    for question in tqdm(questions, desc="Processing questions"):
        try:
            result = get_confidence_and_logprobs(question)
            
            print(f"\nQuestion: {question}")
            print(f"Full response: '{result['full_response']}'")
            print(f"Self-evaluated: {result['self_answer']}. Confidence: {result['self_confidence']}%")
            print(f"LogP-derived: {result['logprob_answer']}. Confidence: {result['logp_confidence']:.1f}%")
            print(f"LogP diff: {result['logp_diff']:.4f}")
            print("-" * 50)
            
            results.append(result)
        except Exception as e:
            print(f"Error processing question: {e}")
            print("-" * 50)
    
    if results:
        # Create separate visualizations with improved formatting
        create_bar_chart_comparison(results)
        create_scatter_correlation(results)
        
        # Generate and print comparison table
        table = create_comparison_table(results)
        print("\nComparison Table:")
        print(table)
        
        # Analyze results
        analysis = analyze_results(results)
        print("\nAnalysis Results:")
        for key, value in analysis.items():
            print(f"{key}: {value}")
        
        # Save results to JSON
        with open('confidence_experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)