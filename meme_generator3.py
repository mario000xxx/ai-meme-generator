import openai
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import os
import sys
import json
import threading
from typing import Optional, Dict, List, Any

# Import PyQt6 components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QComboBox,
    QPushButton, QTextEdit, QFileDialog, QMessageBox, QStatusBar,
    QProgressBar
)
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal

# --- Core AI and Image Logic ---

class AIMemeGenerator:
    """Handles AI text generation and image manipulation for creating memes."""

    def __init__(self, api_key: str):
        """Initializes the generator with an OpenAI API key."""
        if not api_key or api_key == "dummy":
            if api_key != "dummy":
                raise ValueError("API key cannot be empty.")
        # It is highly recommended to use environment variables for API keys
        # rather than hardcoding them.
        if api_key != "dummy":
            self.client = openai.OpenAI(api_key=api_key)
        # A curated list of popular meme templates with descriptions and text positions.
        self.meme_templates = {
            "drake": {
                "url": "https://i.imgflip.com/30b1gx.jpg",
                "text_positions": [(650, 180), (650, 800)],
                "description": "Drake: Disapprove/Approve",
                "json_format": '{"top": "text for top panel", "bottom": "text for bottom panel"}'
            },
            "distracted_boyfriend": {
                "url": "https://i.imgflip.com/1ur9b0.jpg",
                "text_positions": [(300, 50), (50, 500), (550, 50)],
                "description": "Distracted Boyfriend",
                "json_format": '{"boyfriend": "label for person", "girlfriend": "label for responsibility", "new_thing": "label for distraction"}'
            },
            "woman_yelling_cat": {
                "url": "https://i.imgflip.com/345v97.jpg",
                "text_positions": [(140, 150), (600, 150)],
                "description": "Woman Yelling at Cat",
                "json_format": '{"woman": "accusation text", "cat": "confused response"}'
            },
            "two_buttons": {
                "url": "https://i.imgflip.com/1g8my4.jpg",
                "text_positions": [(70, 90), (250, 75), (100, 300)],
                "description": "Two Buttons Choice",
                "json_format": '{"button1": "option 1", "button2": "option 2", "person": "the person struggling"}'
            },
            "change_my_mind": {
                "url": "https://i.imgflip.com/24y43o.jpg",
                "text_positions": [(300, 200)],
                "description": "Change My Mind",
                "json_format": '{"statement": "the controversial statement"}'
            }
        }

    def generate_meme_text(self, topic: str, template: str) -> Dict[str, Any]:
        """
        Generates high-quality meme text using OpenAI GPT-4o with a refined prompt.
        """
        template_info = self.meme_templates[template]

        prompt = f"""
        You are a witty, internet-savvy meme expert. Your goal is to create genuinely funny and clever text for memes.
        Your response MUST be ONLY a valid JSON object.

        **Meme Request:**
        - **Topic:** "{topic}"
        - **Template:** "{template_info['description']}"

        **Instructions:**
        1. Create text that is clever, concise, and highly relevant to the topic.
        2. Aim for observational or satirical humor. Avoid generic jokes.
        3. The text should perfectly fit the emotional context of the template.
        4. Return ONLY a valid JSON object matching the format below.

        **Required JSON Format:**
        {template_info['json_format']}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a meme generator. Create funny, relatable meme text. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating meme text: {e}. Using fallback.")
            return self._get_fallback_text(template, topic)

    def _get_fallback_text(self, template: str, topic: str) -> dict:
        """Provides fallback text if the API call fails."""
        fallbacks = {
            "drake": {"top": f"Regular {topic}", "bottom": f"AI-powered {topic}"},
            "distracted_boyfriend": {"boyfriend": "Me", "girlfriend": "My work", "new_thing": f"Thinking about {topic}"},
            "woman_yelling_cat": {"woman": f"You don't understand {topic}!", "cat": f"Meow?"},
            "two_buttons": {"button1": f"Learn {topic}", "button2": f"Ignore {topic}", "person": "My brain"},
            "change_my_mind": {"statement": f"{topic} is the most important topic. Change my mind."}
        }
        return fallbacks.get(template, {"text": f"A meme about {topic}"})

    def create_meme_image(self, template: str, texts: List[str]) -> Optional[Image.Image]:
        """Downloads a template image and adds the generated text to it."""
        try:
            template_info = self.meme_templates[template]
            response = requests.get(template_info["url"])
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            draw = ImageDraw.Draw(image)

            # Use smaller font for distracted boyfriend template to fit better
            if template == "distracted_boyfriend":
                font_size = int(image.width / 30)
            else:
                font_size = int(image.width / 20)
                
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            positions = template_info["text_positions"]
            for i, text in enumerate(texts):
                if text and i < len(positions):
                    self._draw_text_with_outline(draw, text, positions[i], font, image.width, template)

            return image
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            return None
        except Exception as e:
            print(f"Error creating image: {e}")
            return None

    def create_custom_meme_image(self, image_path: str, text: str) -> Optional[Image.Image]:
        """Creates a meme from a custom uploaded image."""
        try:
            # Load the custom image
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            
            # Calculate font size based on image dimensions
            font_size = max(int(min(image.width, image.height) / 20), 20)
            
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            
            # Position text at the top center and bottom center
            text_positions = [
                (image.width // 2, 50),  # Top center
                (image.width // 2, image.height - 100)  # Bottom center
            ]
            
            # Split text into two parts if it's long, otherwise put it all at top
            words = text.split()
            if len(words) > 3:
                mid_point = len(words) // 2
                texts = [
                    " ".join(words[:mid_point]),
                    " ".join(words[mid_point:])
                ]
            else:
                texts = [text, ""]
            
            for i, text_line in enumerate(texts):
                if text_line and i < len(text_positions):
                    self._draw_centered_text_with_outline(draw, text_line, text_positions[i], font, image.width)
            
            return image
        except Exception as e:
            print(f"Error creating custom meme: {e}")
            return None

    def _draw_centered_text_with_outline(self, draw, text, pos, font, image_width):
        """Draws centered text with outline for custom images."""
        x, y = pos
        
        # Get text dimensions for centering
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        
        # Center the text horizontally
        centered_x = x - (text_width // 2)
        
        # Draw outline
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                if dx != 0 or dy != 0:
                    draw.text((centered_x + dx, y + dy), text, font=font, fill='black')
        
        # Draw main text
        draw.text((centered_x, y), text, font=font, fill='white')

    def _draw_text_with_outline(self, draw, text, pos, font, image_width, template=""):
        """Draws word-wrapped text with a black outline for better readability."""
        x, y = pos
        lines = []
        words = text.split()
        if not words:
            return

        # Adjust text wrapping width for different templates
        if template == "distracted_boyfriend":
            wrap_width = image_width * 0.25  # Narrower text for better fit
        else:
            wrap_width = image_width * 0.45

        current_line = words[0]
        for word in words[1:]:
            if font.getbbox(current_line + " " + word)[2] < wrap_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        line_height = font.getbbox("A")[3] + 4
        for i, line in enumerate(lines):
            line_y = y + (i * line_height)
            # Draw outline
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, line_y + dy), line, font=font, fill='black')
            # Draw text
            draw.text((x, line_y), line, font=font, fill='white')


# --- PyQt6 User Interface ---

class MemeGenerationWorker(QObject):
    """
    A worker QObject that runs the meme generation process in a separate thread.
    Uses signals to communicate results or errors back to the main GUI thread.
    """
    finished = pyqtSignal(object, dict)  # Emits PIL Image and text data
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, generator: AIMemeGenerator, topic: str, template_key: str, custom_image_path: Optional[str] = None):
        super().__init__()
        self.generator = generator
        self.topic = topic
        self.template_key = template_key
        self.custom_image_path = custom_image_path

    def run(self):
        """The main work of the thread."""
        try:
            if self.template_key == "custom":
                # For custom images, create simple text overlay
                self.progress.emit("1/2: Loading custom image...")
                image = self.generator.create_custom_meme_image(self.custom_image_path, self.topic)
                if not image:
                    self.error.emit("Failed to load the custom image.")
                    return
                self.progress.emit("2/2: Adding text...")
                # Create simple text data for custom images
                text_data = {"custom_text": self.topic}
            else:
                self.progress.emit("1/3: Generating meme text with GPT-4o...")
                text_data = self.generator.generate_meme_text(self.topic, self.template_key)

                self.progress.emit("2/3: Downloading template and adding text...")
                texts = list(text_data.values())
                image = self.generator.create_meme_image(self.template_key, texts)

                if not image:
                    self.error.emit("Failed to create the meme image. Check console for details.")
                    return

                self.progress.emit("3/3: Finalizing...")
            
            self.finished.emit(image, text_data)
        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {e}")

class PyQtMemeApp(QMainWindow):
    """A PyQt6 GUI for the AI Meme Generator."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Meme Generator")
        self.setGeometry(100, 100, 850, 650)
        self.setMinimumSize(600, 500)

        self.generator: Optional[AIMemeGenerator] = None
        self.current_meme_image: Optional[Image.Image] = None

        self.create_widgets()
        self.create_layouts()
        self.create_connections()
        self.check_for_api_key()

    def create_widgets(self):
        """Creates all the widgets for the application."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # --- Controls Group ---
        self.controls_group = QGroupBox("Controls")
        self.api_key_label = QLabel("API Key:")
        self.api_key_entry = QLineEdit()
        self.api_key_entry.setEchoMode(QLineEdit.EchoMode.Password)
        self.topic_label = QLabel("Topic:")
        self.topic_entry = QLineEdit("learning a new programming language")
        self.template_label = QLabel("Template:")
        self.template_combo = QComboBox()
        self.populate_templates()
        
        # Custom image upload section
        self.custom_image_label = QLabel("Custom Image:")
        self.upload_btn = QPushButton("Upload Image")
        self.custom_image_path = None
        self.custom_image_display = QLabel("No custom image selected")
        self.custom_image_display.setStyleSheet("color: gray; font-style: italic;")
        
        self.generate_btn = QPushButton("Generate Meme")

        # --- Output Group ---
        self.output_group = QGroupBox("Output")
        self.image_label = QLabel("\nYour generated meme\nwill appear here.\n")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc;")
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.save_btn = QPushButton("Save Image...")
        self.copy_btn = QPushButton("Copy Text")

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_layouts(self):
        """Sets up the layout of the widgets."""
        main_layout = QVBoxLayout(self.central_widget)

        # Controls layout
        controls_layout = QGridLayout(self.controls_group)
        controls_layout.addWidget(self.api_key_label, 0, 0)
        controls_layout.addWidget(self.api_key_entry, 0, 1)
        controls_layout.addWidget(self.topic_label, 1, 0)
        controls_layout.addWidget(self.topic_entry, 1, 1)
        controls_layout.addWidget(self.template_label, 2, 0)
        controls_layout.addWidget(self.template_combo, 2, 1)
        controls_layout.addWidget(self.custom_image_label, 3, 0)
        custom_image_layout = QHBoxLayout()
        custom_image_layout.addWidget(self.upload_btn)
        custom_image_layout.addWidget(self.custom_image_display)
        controls_layout.addLayout(custom_image_layout, 3, 1)
        controls_layout.addWidget(self.generate_btn, 4, 1)

        # Output layout
        output_layout = QHBoxLayout(self.output_group)
        right_panel_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.copy_btn)

        right_panel_layout.addWidget(self.text_display)
        right_panel_layout.addLayout(button_layout)
        
        output_layout.addWidget(self.image_label, 3) # 3/5 of space
        output_layout.addLayout(right_panel_layout, 2) # 2/5 of space

        main_layout.addWidget(self.controls_group)
        main_layout.addWidget(self.output_group)

    def create_connections(self):
        """Connects widget signals to corresponding slots."""
        self.generate_btn.clicked.connect(self.start_generation_thread)
        self.save_btn.clicked.connect(self.save_meme)
        self.copy_btn.clicked.connect(self.copy_text)
        self.upload_btn.clicked.connect(self.upload_custom_image)
        self.template_combo.currentTextChanged.connect(self.on_template_changed)
        
        self.set_ui_state_ready()

    def check_for_api_key(self):
        """Checks for OpenAI API key in environment variables on startup."""
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.api_key_entry.setText(api_key)
            self.update_status("API Key loaded from environment.")

    def populate_templates(self):
        """Populates the template dropdown menu."""
        # Use a dummy key just to get the template list
        dummy_gen = AIMemeGenerator(api_key="dummy")
        for key, value in dummy_gen.meme_templates.items():
            self.template_combo.addItem(f"{value['description']} ({key})", key)
        
        # Add custom image option
        self.template_combo.addItem("Custom Image", "custom")

    def upload_custom_image(self):
        """Opens a dialog to upload a custom image for memeing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Image",
            "",
            "Image files (*.png *.jpg *.jpeg *.gif *.bmp)"
        )
        
        if file_path:
            self.custom_image_path = file_path
            filename = os.path.basename(file_path)
            self.custom_image_display.setText(f"Selected: {filename}")
            self.custom_image_display.setStyleSheet("color: green;")
            # Auto-select custom template when image is uploaded
            custom_index = self.template_combo.findData("custom")
            if custom_index >= 0:
                self.template_combo.setCurrentIndex(custom_index)

    def on_template_changed(self):
        """Called when template selection changes."""
        current_template = self.template_combo.currentData()
        if current_template == "custom":
            self.upload_btn.setEnabled(True)
            if not self.custom_image_path:
                self.custom_image_display.setText("Please upload an image")
                self.custom_image_display.setStyleSheet("color: red;")
        else:
            self.upload_btn.setEnabled(True)  # Always allow uploading

    def start_generation_thread(self):
        """Handles meme generation in a separate thread to keep the UI responsive."""
        api_key = self.api_key_entry.text().strip()
        if not api_key:
            QMessageBox.critical(self, "API Key Missing", "Please enter your OpenAI API key.")
            return

        # Check if custom template is selected and image is uploaded
        template_key = self.template_combo.currentData()
        if template_key == "custom" and not self.custom_image_path:
            QMessageBox.critical(self, "Custom Image Missing", "Please upload a custom image first.")
            return

        try:
            self.generator = AIMemeGenerator(api_key)
        except ValueError as e:
            QMessageBox.critical(self, "Initialization Error", str(e))
            return
            
        self.set_ui_state_busy()

        topic = self.topic_entry.text().strip()
        
        # Create a QThread and a worker
        self.thread = QThread()
        self.worker = MemeGenerationWorker(self.generator, topic, template_key, self.custom_image_path)
        self.worker.moveToThread(self.thread)

        # Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.progress.connect(self.update_status)
        
        # Clean up after the worker is done
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def display_results(self, image: Image.Image, text_data: Dict):
        """Updates the UI with the generated meme and text."""
        self.current_meme_image = image

        # Convert PIL Image to QPixmap
        q_image = QImage(image.tobytes(), image.width, image.height, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Scale pixmap to fit the label while preserving aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)

        self.image_label.setPixmap(scaled_pixmap)
        
        # Display text in a user-friendly format instead of JSON
        self.display_text_friendly(text_data)
        
        self.set_ui_state_ready()
        self.update_status("Meme generated successfully!")

    def display_text_friendly(self, text_data: Dict):
        """Displays the meme text in a user-friendly format instead of JSON."""
        display_text = "Generated Meme Text:\n\n"
        
        # Create readable labels for different templates
        label_mappings = {
            "top": "Top Text",
            "bottom": "Bottom Text", 
            "boyfriend": "Person",
            "girlfriend": "Current Responsibility",
            "new_thing": "New Distraction",
            "woman": "Accusation",
            "cat": "Response",
            "button1": "Option 1",
            "button2": "Option 2", 
            "person": "Person Deciding",
            "statement": "Statement",
            "custom_text": "Custom Text"
        }
        
        for key, value in text_data.items():
            friendly_label = label_mappings.get(key, key.title())
            display_text += f"{friendly_label}: {value}\n"
        
        self.text_display.setText(display_text)
        
    def save_meme(self):
        """Opens a dialog to save the current meme image."""
        if not self.current_meme_image:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Meme",
            "my-awesome-meme.png",
            "PNG files (*.png);;JPEG files (*.jpg)"
        )

        if file_path:
            try:
                self.current_meme_image.save(file_path)
                QMessageBox.information(self, "Success", f"Meme saved to {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image: {e}")

    def copy_text(self):
        """Copies the generated text to the clipboard."""
        text = self.text_display.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.update_status("Copied text to clipboard!")

    def update_status(self, message: str):
        """Updates the text in the status bar."""
        self.status_label.setText(message)

    def set_ui_state_busy(self):
        """Disables UI elements during processing."""
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0,0) # Indeterminate mode

    def set_ui_state_ready(self):
        """Enables UI elements when processing is complete."""
        self.generate_btn.setEnabled(True)
        self.save_btn.setEnabled(self.current_meme_image is not None)
        self.copy_btn.setEnabled(self.text_display.toPlainText() != "")
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0,100)

    def handle_error(self, message: str):
        """Displays an error message and resets the UI."""
        QMessageBox.critical(self, "Error", message)
        self.set_ui_state_ready()
        self.update_status("Ready.")

    def closeEvent(self, event):
        """Ensures the background thread is terminated if running when the window closes."""
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait() # Wait for the thread to finish
        event.accept()

# --- Main Execution ---
def main():
    """Main function to check dependencies and run the GUI."""
    try:
        __import__('PyQt6')
    except ImportError:
        # A simple fallback message if PyQt6 isn't available
        root_tk = __import__('tkinter').Tk()
        root_tk.withdraw()
        QMessageBox.critical(None, "Missing Packages",
                             "PyQt6 is required. Please run:\n\n"
                             "pip install PyQt6")
        return

    app = QApplication(sys.argv)
    window = PyQtMemeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()