import sys
import streamlit.web.cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "demo_app/demo_app.py", "--server.port=80", "--server.enableXsrfProtection=false"]
    sys.exit(stcli.main())