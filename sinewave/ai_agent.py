import os
import openai
from dotenv import load_dotenv
import pandas as pd
import json
from datetime import datetime

class MaintenanceAIAgent:
    """
    AI Agent for maintenance analysis and chat interface using GPT-4
    """
    def __init__(self):
        """
        Initialize the AI agent with OpenAI configuration
        """
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.conversation_history = []
        
    def generate_maintenance_report(self, predictor_output, equipment_data):
        """
        Generate a detailed maintenance report using GPT-4
        
        Args:
            predictor_output (dict): Output from EquipmentPredictor
            equipment_data (pd.DataFrame): Equipment sensor data
            
        Returns:
            str: Detailed maintenance report
        """
        # Prepare context for GPT-4
        context = {
            'failure_probs': predictor_output['failure_probabilities'].tolist(),
            'anomaly_scores': predictor_output['anomaly_scores'].tolist(),
            'equipment_metrics': equipment_data.describe().to_dict()
        }
        
        prompt = f"""
        As a maintenance expert, analyze the following equipment data and generate a detailed report:
        
        Equipment Data Summary:
        {json.dumps(context, indent=2)}
        
        Please provide:
        1. Overall equipment health assessment
        2. Identified risks and anomalies
        3. Recommended maintenance actions
        4. Priority level for each recommendation
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert maintenance analyst providing detailed equipment analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def chat_interaction(self, user_message, predictor_output=None):
        """
        Handle chat interactions with users
        
        Args:
            user_message (str): User's input message
            predictor_output (dict, optional): Latest predictor output
            
        Returns:
            str: AI response
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Prepare system context
        system_context = """
        You are an AI maintenance assistant helping users understand equipment health and maintenance needs.
        You have access to real-time equipment data and can provide insights about maintenance requirements.
        """
        
        # Include predictor output if available
        if predictor_output:
            system_context += f"\nCurrent equipment status: {json.dumps(predictor_output, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_context},
            *self.conversation_history
        ]
        
        # Get response from GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        
        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        
        return ai_response
    
    def save_conversation(self, filepath):
        """
        Save conversation history to file
        
        Args:
            filepath (str): Path to save conversation
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_data = {
            "timestamp": timestamp,
            "conversation": self.conversation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(conversation_data, f, indent=2)
    
    def clear_conversation(self):
        """
        Clear conversation history
        """
        self.conversation_history = []
