# styles.py
NAV_BAR_HTML = """
    <style>
        header[data-testid="stHeader"] {
            display: none !important;
        }
        
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        .full-width-nav {
            background-color: #046b21;
            width: 100vw;
            height: 70px;
            display: flex;
            align-items: center;
            padding: 0 50px;
            margin: 0;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            position: fixed;
            top: 0;
            left: 0;
            z-index: 9999;
        }
        
        .nav-text {
            color: white;
            font-size: 24px;
            font-weight: 800;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-left: 15px;
        }

        .content-wrapper {
            margin-top: 100px;
            padding: 0 50px;
        }
    </style>
    
    <div class="full-width-nav">
        <img src="https://raw.githubusercontent.com/metrych-creator/bitcoin_prediction/refs/heads/main/view/static/images/logo.png" height="100" width="140" box-shadow: 18px 24px 24px 0px rgba(0, 0, 0, 1)>
    </div>"""
   