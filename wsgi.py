from app.main import app 
import app.util as util
if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run()