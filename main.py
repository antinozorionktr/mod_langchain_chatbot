import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import re
import json
import chromadb
from datetime import datetime
import hashlib
import requests

class VectorDatabaseManager:
    """Manages vector database operations for semantic search across tables"""
    
    def __init__(self, embedding_model="nomic-embed-text", persist_directory="./vector_db"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.persist_directory = persist_directory
        self.vectorstore = None
        
    def create_documents_from_data(self, engine):
        """Create documents from database tables for vector storage"""
        documents = []
        
        # Define table mappings with their descriptions
        table_configs = {
            'e_sit_rep': {
                'description': 'Electronic Situation Reports - incident reports with location, time, and details',
                'key_fields': ['incident_date', 'incident_type', 'incident_subtype', 'description', 'location_info']
            },
            'en_activity': {
                'description': 'Enemy Activity reports - sensor-detected activities and targets',
                'key_fields': ['sensor_type', 'tgt_type', 'activity_type', 'description', 'location_info']
            },
            'imint': {
                'description': 'Imagery Intelligence reports - visual intelligence with target and activity analysis',
                'key_fields': ['date', 'tgt_type', 'activity_type', 'incident_type', 'description', 'grading']
            },
            'tac_int': {
                'description': 'Tactical Intelligence reports - field intelligence with target and activity details',
                'key_fields': ['date', 'tgt_type', 'activity_type', 'incident_type', 'description', 'grading']
            },
            'ecas': {
                'description': 'Electronic Counter Attack System data - electronic warfare activities and emitters',
                'key_fields': ['date', 'emitter_type', 'emitter_name', 'location', 'description']
            }
        }
        
        try:
            for table_name, config in table_configs.items():
                # Get data from each table
                query = f"""
                SELECT * FROM {table_name} 
                WHERE description IS NOT NULL 
                ORDER BY created_at DESC 
                LIMIT 1000
                """
                
                df = pd.read_sql(query, engine)
                
                if not df.empty:
                    for _, row in df.iterrows():
                        # Create comprehensive document content
                        content_parts = [
                            f"Table: {table_name}",
                            f"Description: {config['description']}",
                            f"Record ID: {row.get('id', 'N/A')}"
                        ]
                        
                        # Add key field information
                        for field in config['key_fields']:
                            if field in row and pd.notna(row[field]):
                                if field == 'location_info':
                                    # Combine location fields
                                    location_parts = []
                                    for loc_field in ['lat', 'long', 'incident_state', 'location']:
                                        if loc_field in row and pd.notna(row[loc_field]):
                                            location_parts.append(f"{loc_field}: {row[loc_field]}")
                                    if location_parts:
                                        content_parts.append(f"Location: {', '.join(location_parts)}")
                                else:
                                    content_parts.append(f"{field}: {row[field]}")
                        
                        # Add organizational information
                        org_fields = ['cmd_name', 'corps_name', 'div_name', 'bde_name', 'unit_name']
                        org_info = []
                        for field in org_fields:
                            if field in row and pd.notna(row[field]):
                                org_info.append(f"{field}: {row[field]}")
                        if org_info:
                            content_parts.append(f"Organization: {', '.join(org_info)}")
                        
                        # Create metadata
                        metadata = {
                            'table': table_name,
                            'id': str(row.get('id', '')),
                            'type': config['description'],
                            'date': str(row.get('date', row.get('incident_date', row.get('created_at', '')))),
                        }
                        
                        # Add type information to metadata
                        if 'incident_type' in row and pd.notna(row['incident_type']):
                            metadata['incident_type'] = str(row['incident_type'])
                        if 'activity_type' in row and pd.notna(row['activity_type']):
                            metadata['activity_type'] = str(row['activity_type'])
                        if 'tgt_type' in row and pd.notna(row['tgt_type']):
                            metadata['target_type'] = str(row['tgt_type'])
                        
                        # Create document
                        doc = Document(
                            page_content='\n'.join(content_parts),
                            metadata=metadata
                        )
                        documents.append(doc)
                        
        except Exception as e:
            st.error(f"Error creating documents from table {table_name}: {e}")
        
        return documents
    
    def build_vector_database(self, engine):
        """Build and persist vector database from all tables"""
        try:
            st.info("Building vector database from intelligence data...")
            
            # Create documents from database
            documents = self.create_documents_from_data(engine)
            
            if not documents:
                st.error("No documents created from database")
                return False
            
            # Split documents if they're too large
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            st.success(f"Vector database built successfully with {len(split_docs)} document chunks!")
            return True
            
        except Exception as e:
            st.error(f"Error building vector database: {e}")
            return False
    
    def load_vector_database(self):
        """Load existing vector database"""
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                return True
            return False
        except Exception as e:
            st.error(f"Error loading vector database: {e}")
            return False
    
    def semantic_search(self, query, k=5):
        """Perform semantic search across all intelligence data"""
        if not self.vectorstore:
            return []
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            st.error(f"Error in semantic search: {e}")
            return []

class EnhancedPostgreSQLChatbot:
    def __init__(self, db_url, llm_model="llama3.1", embedding_model="nomic-embed-text", ollama_base_url="http://localhost:11434"):
        """
        Initialize the enhanced PostgreSQL chatbot with local Llama model
        """
        self.db_url = db_url
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        
        # Initialize database connection
        self.engine = create_engine(db_url)
        self.db = SQLDatabase.from_uri(db_url)
        
        # Initialize local Llama LLM with Ollama
        self.llm = Ollama(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0.1,
            top_p=0.9,
            num_predict=2048,
            verbose=True
        )
        
        # Initialize vector database manager with local embeddings
        self.vector_manager = VectorDatabaseManager(embedding_model)
        
        # Initialize SQL agent with custom prompt for Llama
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Custom system prompt for Llama SQL agent
        self.sql_agent_prefix = """
        You are an expert SQL analyst working with a PostgreSQL database containing intelligence data.
        
        Database Schema:
        - e_sit_rep: Electronic situation reports with incident data
        - en_activity: Enemy activity reports with sensor data  
        - imint: Imagery intelligence reports
        - tac_int: Tactical intelligence reports
        - ecas: Electronic counter-attack system data
        
        Guidelines:
        1. Always write PostgreSQL-compatible SQL queries
        2. Use proper table and column names from the schema
        3. Include relevant JOINs when analyzing across tables
        4. Limit results to reasonable numbers (typically 50-100 rows)
        5. Use proper date/time formatting for PostgreSQL
        6. Provide clear explanations of your analysis
        7. If you need to count or aggregate, use appropriate GROUP BY clauses
        
        Always start your response with the SQL query in a code block, then provide analysis.
        """
        
        try:
            self.agent = create_sql_agent(
                llm=self.llm,
                toolkit=self.toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                max_iterations=3,
                max_execution_time=120,
                early_stopping_method="generate",
                prefix=self.sql_agent_prefix
            )
        except Exception as e:
            st.error(f"Error creating SQL agent: {e}")
            self.agent = None
        
        # Initialize conversation history
        self.chat_history = []
        
        # Load or build vector database
        self.setup_vector_database()
    
    def check_ollama_connection(self):
        """Check if Ollama server is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                return True, model_names
            else:
                return False, []
        except Exception as e:
            return False, []
    
    def setup_vector_database(self):
        """Setup vector database - load existing or build new"""
        if not self.vector_manager.load_vector_database():
            st.info("No existing vector database found. Building new one...")
            return self.vector_manager.build_vector_database(self.engine)
        else:
            st.success("Loaded existing vector database!")
            return True
    
    def rebuild_vector_database(self):
        """Force rebuild of vector database"""
        return self.vector_manager.build_vector_database(self.engine)
    
    def get_related_context(self, query):
        """Get related context from vector database"""
        docs = self.vector_manager.semantic_search(query, k=3)
        
        if not docs:
            return ""
        
        context_parts = []
        for doc in docs:
            context_parts.append(f"Related Information:\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def enhanced_chat(self, user_query):
        """
        Enhanced chat function with vector database context and local Llama
        """
        try:
            # Get semantic context from vector database
            vector_context = self.get_related_context(user_query)
            
            # For simple queries, try direct SQL generation
            if self.agent is None:
                return self.fallback_query_handler(user_query, vector_context)
            
            # Create enhanced query with context for Llama
            enhanced_query = f"""
            Context from Intelligence Database:
            {vector_context}
            
            User Question: {user_query}
            
            Instructions:
            1. Analyze the user question and provided context
            2. Write appropriate PostgreSQL SQL queries to answer the question
            3. Execute the queries to get results
            4. Provide comprehensive analysis combining SQL results and contextual information
            5. Focus on intelligence analysis and insights
            
            Database contains intelligence data from multiple sources:
            - e_sit_rep: Electronic situation reports
            - en_activity: Enemy activity reports  
            - imint: Imagery intelligence
            - tac_int: Tactical intelligence
            - ecas: Electronic warfare data
            """
            
            # Get response from SQL agent
            response = self.agent.invoke({"input": enhanced_query})
            
            # Extract response text
            if hasattr(response, 'output'):
                response_text = response.output
            elif isinstance(response, dict) and 'output' in response:
                response_text = response['output']
            else:
                response_text = str(response)
            
            # Extract SQL and create visualization
            sql_query = self.extract_sql_from_response(response_text)
            visualization = None
            data_df = pd.DataFrame()
            
            if sql_query:
                try:
                    data_df = self.execute_query(sql_query)
                    if not data_df.empty and len(data_df) > 0:
                        chart_type = self.determine_chart_type(data_df, user_query)
                        if chart_type and chart_type != 'table':
                            visualization = self.create_visualization(data_df, chart_type, user_query)
                except Exception as e:
                    st.warning(f"Could not create visualization: {e}")
            
            # Add vector search results to response if relevant
            if vector_context:
                related_docs = self.vector_manager.semantic_search(user_query, k=3)
                if related_docs:
                    response_text += "\n\n### 🔍 Related Intelligence Records:\n"
                    for i, doc in enumerate(related_docs, 1):
                        metadata = doc.metadata
                        response_text += f"\n{i}. **{metadata.get('table', 'Unknown')}** (ID: {metadata.get('id', 'N/A')})\n"
                        response_text += f"   Date: {metadata.get('date', 'N/A')}\n"
                        response_text += f"   Summary: {doc.page_content[:200]}...\n"
            
            # Store in chat history
            self.chat_history.append({"user": user_query, "assistant": response_text})
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            return {
                'response': response_text,
                'data': data_df,
                'visualization': visualization,
                'sql_query': sql_query,
                'vector_context': vector_context
            }
            
        except Exception as e:
            error_msg = f"Error processing enhanced query: {str(e)}"
            st.error(error_msg)
            return {
                'response': error_msg,
                'data': pd.DataFrame(),
                'visualization': None,
                'sql_query': None,
                'vector_context': ""
            }
    
    def fallback_query_handler(self, user_query, vector_context):
        """Fallback handler when SQL agent fails"""
        try:
            # Simple prompt for direct SQL generation
            prompt = f"""
            Given this user question about intelligence data: {user_query}
            
            Context: {vector_context}
            
            Generate a PostgreSQL SQL query to answer this question. The database has tables:
            - e_sit_rep (incident reports)
            - en_activity (enemy activity)
            - imint (imagery intelligence)
            - tac_int (tactical intelligence) 
            - ecas (electronic warfare)
            
            Return only the SQL query, no explanation.
            """
            
            sql_response = self.llm.invoke(prompt)
            sql_query = self.extract_sql_from_response(sql_response)
            
            if sql_query:
                data_df = self.execute_query(sql_query)
                visualization = None
                
                if not data_df.empty:
                    chart_type = self.determine_chart_type(data_df, user_query)
                    if chart_type and chart_type != 'table':
                        visualization = self.create_visualization(data_df, chart_type, user_query)
                
                # Generate analysis
                analysis_prompt = f"""
                Based on this SQL query result for the question "{user_query}":
                
                Query: {sql_query}
                Results: {data_df.head(10).to_string() if not data_df.empty else "No results"}
                
                Provide a comprehensive analysis of these intelligence findings.
                """
                
                analysis = self.llm.invoke(analysis_prompt)
                
                return {
                    'response': analysis,
                    'data': data_df,
                    'visualization': visualization,
                    'sql_query': sql_query,
                    'vector_context': vector_context
                }
            else:
                return {
                    'response': "I couldn't generate a proper SQL query for your question. Please try rephrasing.",
                    'data': pd.DataFrame(),
                    'visualization': None,
                    'sql_query': None,
                    'vector_context': vector_context
                }
                
        except Exception as e:
            return {
                'response': f"Error in fallback handler: {str(e)}",
                'data': pd.DataFrame(),
                'visualization': None,
                'sql_query': None,
                'vector_context': ""
            }
    
    def execute_query(self, query):
        """Execute SQL query and return results"""
        try:
            query = query.strip().rstrip(';')
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            st.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def determine_chart_type(self, df, user_query):
        """Determine the best chart type based on data and user query"""
        if df.empty or len(df) == 0:
            return None
            
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        query_lower = user_query.lower()
        
        # Chart type determination logic
        if any(word in query_lower for word in ['trend', 'time', 'over time', 'timeline', 'monthly', 'daily', 'yearly']):
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                return 'line'
            elif len(df.columns) >= 2 and len(numeric_cols) > 0:
                return 'line'
        elif any(word in query_lower for word in ['compare', 'comparison', 'bar', 'category', 'top', 'bottom']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                return 'bar'
        elif any(word in query_lower for word in ['distribution', 'histogram', 'frequency']):
            if len(numeric_cols) > 0:
                return 'histogram'
        elif any(word in query_lower for word in ['pie', 'proportion', 'percentage', 'share']):
            if len(categorical_cols) > 0 and len(numeric_cols) > 0 and len(df) <= 10:
                return 'pie'
        elif any(word in query_lower for word in ['scatter', 'correlation', 'relationship', 'vs']):
            if len(numeric_cols) >= 2:
                return 'scatter'
        
        # Default logic based on data structure
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            if len(df) <= 20:
                return 'bar'
        elif len(numeric_cols) >= 2:
            return 'scatter'
        elif len(datetime_cols) > 0 and len(numeric_cols) > 0:
            return 'line'
        elif len(numeric_cols) > 0:
            return 'histogram'
        else:
            return 'table'
    
    def create_visualization(self, df, chart_type, user_query):
        """Create visualization based on chart type and data"""
        if df.empty or chart_type == 'table' or len(df) == 0:
            return None
            
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            if chart_type == 'bar':
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    x_col = categorical_cols[0]
                    y_col = numeric_cols[0]
                    
                    if len(df) > 20:
                        df_plot = df.nlargest(20, y_col)
                    else:
                        df_plot = df
                    
                    fig = px.bar(
                        df_plot, 
                        x=x_col, 
                        y=y_col, 
                        title=f'{y_col} by {x_col}',
                        color=y_col,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    return fig
                    
            elif chart_type == 'line':
                if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                    x_col = datetime_cols[0]
                    y_col = numeric_cols[0]
                    fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} over {x_col}')
                    return fig
                elif len(df.columns) >= 2 and len(numeric_cols) > 0:
                    x_col = df.columns[0]
                    y_col = numeric_cols[0]
                    fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} trend')
                    return fig
                    
            elif chart_type == 'scatter':
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                    
                    color_col = None
                    if len(categorical_cols) > 0:
                        color_col = categorical_cols[0]
                    
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f'{y_col} vs {x_col}'
                    )
                    return fig
                    
            elif chart_type == 'pie':
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    names_col = categorical_cols[0]
                    values_col = numeric_cols[0]
                    
                    if len(df) > 10:
                        df_plot = df.nlargest(10, values_col)
                    else:
                        df_plot = df
                    
                    fig = px.pie(
                        df_plot, 
                        names=names_col, 
                        values=values_col, 
                        title=f'Distribution of {values_col} by {names_col}'
                    )
                    return fig
                    
            elif chart_type == 'histogram':
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    fig = px.histogram(
                        df, 
                        x=col, 
                        title=f'Distribution of {col}',
                        nbins=min(30, len(df.unique()) if hasattr(df[col], 'unique') else 30)
                    )
                    return fig
                    
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            
        return None
    
    def extract_sql_from_response(self, response):
        """Extract SQL query from agent response"""
        sql_patterns = [
            r'```sql\n(.*?)\n```',
            r'```\n(SELECT.*?)\n```',
            r'```\n(WITH.*?)\n```',
            r'(SELECT.*?)(?:\n\n|\nFinal|\nI hope|$)',
            r'(WITH.*?)(?:\n\n|\nFinal|\nI hope|$)',
        ]
        
        response_str = str(response)
        
        for pattern in sql_patterns:
            match = re.search(pattern, response_str, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                sql = re.sub(r'\n+', ' ', sql)
                sql = re.sub(r'\s+', ' ', sql)
                return sql.strip()
        
        return None

def main():
    st.set_page_config(
        page_title="Local Llama Intelligence Chatbot",
        page_icon="🦙",
        layout="wide"
    )
    
    st.title("🦙 Local Llama Intelligence Database Chatbot")
    st.markdown("AI-powered analysis using local Llama models via Ollama")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ollama Configuration
        st.subheader("🦙 Ollama Configuration")
        ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
        
        # Check Ollama connection and get available models
        if st.button("🔍 Check Available Models"):
            try:
                response = requests.get(f"{ollama_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    if models:
                        st.success("✅ Ollama connected successfully!")
                        st.write("Available models:")
                        for model in models:
                            st.write(f"- {model['name']}")
                    else:
                        st.warning("Ollama connected but no models found. Please install models first.")
                        st.code("ollama pull llama3.1")
                        st.code("ollama pull nomic-embed-text")
                else:
                    st.error("❌ Cannot connect to Ollama")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
                st.info("Make sure Ollama is running: `ollama serve`")
        
        # Model selection
        llm_model = st.text_input("LLM Model", value="llama3.1", help="e.g., llama3.1, llama2, codellama")
        embedding_model = st.text_input("Embedding Model", value="nomic-embed-text", help="Model for vector embeddings")
        
        # Database connection
        st.subheader("🗄️ Database Connection")
        db_host = st.text_input("Host", value="localhost")
        db_port = st.text_input("Port", value="5432")
        db_name = st.text_input("Database Name")
        db_user = st.text_input("Username")
        db_password = st.text_input("Password", type="password")
        
        # Connect button
        if st.button("🚀 Connect & Initialize"):
            if all([db_host, db_port, db_name, db_user, db_password]):
                db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                try:
                    # Test database connection
                    test_engine = create_engine(db_url)
                    test_connection = test_engine.connect()
                    test_connection.close()
                    
                    # Test Ollama connection
                    ollama_response = requests.get(f"{ollama_url}/api/tags")
                    if ollama_response.status_code != 200:
                        st.error("❌ Cannot connect to Ollama. Make sure it's running.")
                        return
                    
                    # Initialize enhanced chatbot
                    with st.spinner("Initializing local Llama chatbot..."):
                        chatbot = EnhancedPostgreSQLChatbot(
                            db_url, 
                            llm_model=llm_model,
                            embedding_model=embedding_model,
                            ollama_base_url=ollama_url
                        )
                        st.session_state.chatbot = chatbot
                    
                    st.success("✅ Connected successfully with local Llama!")
                    
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
            else:
                st.error("Please fill in all required fields")
        
        # Rebuild vector database button
        if 'chatbot' in st.session_state:
            if st.button("🔄 Rebuild Vector Database"):
                with st.spinner("Rebuilding vector database..."):
                    if st.session_state.chatbot.rebuild_vector_database():
                        st.success("Vector database rebuilt successfully!")
                    else:
                        st.error("Failed to rebuild vector database")
    
    # Main interface
    if 'chatbot' in st.session_state:
        chatbot = st.session_state.chatbot
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("🦙 Local LLM Active")
        with col2:
            st.success("🔍 Vector Search Enabled")
        with col3:
            st.success("🗄️ Database Connected")
    
        # Chat interface
        st.header("💬 Chat with Your Intelligence Database")
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(message["user"])
                
                with st.chat_message("assistant"):
                    st.write(message["assistant"])
                    
                    # Display additional data if available
                    if "data" in message and not message["data"].empty:
                        with st.expander("📊 Query Results"):
                            st.dataframe(message["data"])
                    
                    if "visualization" in message and message["visualization"]:
                        with st.expander("📈 Visualization"):
                            st.plotly_chart(message["visualization"], use_container_width=True)
                    
                    if "sql_query" in message and message["sql_query"]:
                        with st.expander("🔍 SQL Query"):
                            st.code(message["sql_query"], language="sql")
        
        # Chat input
        user_input = st.chat_input("Ask about your intelligence data...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"user": user_input, "assistant": "Processing..."})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(user_input)
            
            # Process the query
            with st.chat_message("assistant"):
                with st.spinner("🦙 Llama is thinking..."):
                    try:
                        result = chatbot.enhanced_chat(user_input)
                        
                        # Display response
                        st.write(result['response'])
                        
                        # Display data table if available
                        if not result['data'].empty:
                            with st.expander("📊 Query Results", expanded=True):
                                st.dataframe(result['data'])
                        
                        # Display visualization if available
                        if result['visualization']:
                            with st.expander("📈 Visualization", expanded=True):
                                st.plotly_chart(result['visualization'], use_container_width=True)
                        
                        # Display SQL query if available
                        if result['sql_query']:
                            with st.expander("🔍 Generated SQL Query"):
                                st.code(result['sql_query'], language="sql")
                        
                        # Display vector context if available
                        if result['vector_context']:
                            with st.expander("🔍 Related Context"):
                                st.text(result['vector_context'])
                        
                        # Update chat history with actual response
                        st.session_state.chat_history[-1] = {
                            "user": user_input,
                            "assistant": result['response'],
                            "data": result['data'],
                            "visualization": result['visualization'],
                            "sql_query": result['sql_query']
                        }
                        
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history[-1]["assistant"] = error_msg
        
        # Quick query examples
        st.header("💡 Example Queries")
        example_queries = [
            "Show me recent incident reports by type",
            "What are the most common enemy activities detected?",
            "Analyze trends in intelligence reports over time",
            "Show me electronic warfare activities by location",
            "What targets are most frequently reported?",
            "Compare activity levels between different sensor types",
            "Show me incidents in specific geographic areas",
            "What are the recent tactical intelligence findings?"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            col = cols[i % 2]
            if col.button(query, key=f"example_{i}"):
                # Trigger the query by setting it as user input
                st.session_state.example_query = query
                st.rerun()
        
        # Handle example query execution
        if 'example_query' in st.session_state:
            example_query = st.session_state.example_query
            del st.session_state.example_query
            
            # Add to chat history
            st.session_state.chat_history.append({"user": example_query, "assistant": "Processing..."})
            
            # Process the example query
            with st.spinner("🦙 Processing example query..."):
                try:
                    result = chatbot.enhanced_chat(example_query)
                    
                    # Update chat history with result
                    st.session_state.chat_history[-1] = {
                        "user": example_query,
                        "assistant": result['response'],
                        "data": result['data'],
                        "visualization": result['visualization'],
                        "sql_query": result['sql_query']
                    }
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing example query: {str(e)}")
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Database statistics
        with st.expander("📊 Database Statistics"):
            try:
                tables = ['e_sit_rep', 'en_activity', 'imint', 'tac_int', 'ecas']
                stats_data = []
                
                for table in tables:
                    try:
                        count_query = f"SELECT COUNT(*) as count FROM {table}"
                        result = pd.read_sql(count_query, chatbot.engine)
                        count = result['count'].iloc[0]
                        stats_data.append({"Table": table, "Records": count})
                    except Exception as e:
                        stats_data.append({"Table": table, "Records": f"Error: {str(e)}"})
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error fetching database statistics: {e}")
    
    else:
        # Welcome screen
        st.info("👈 Please configure and connect to your database using the sidebar to get started.")
        
        st.markdown("""
        ## 🚀 Getting Started
        
        ### Prerequisites:
        1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
        2. **Start Ollama**: Run `ollama serve` in terminal
        3. **Install Models**:
           ```bash
           ollama pull llama3.1
           ollama pull nomic-embed-text
           ```
        
        ### Features:
        - 🦙 **Local LLM**: Uses Llama models running locally via Ollama
        - 🔍 **Vector Search**: Semantic search across intelligence data
        - 📊 **Auto Visualization**: Automatic chart generation
        - 💬 **Natural Language**: Ask questions in plain English
        - 🗄️ **PostgreSQL**: Direct database integration
        
        ### Database Schema:
        - `e_sit_rep`: Electronic situation reports
        - `en_activity`: Enemy activity reports
        - `imint`: Imagery intelligence
        - `tac_int`: Tactical intelligence
        - `ecas`: Electronic counter-attack system data
        """)

if __name__ == "__main__":
    main()