from app.main import app 
import app.util 
if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run()