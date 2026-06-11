#!/usr/bin/env python3
"""
Flask Application Entry Point for ToF Gesture Classification System

Run with: python run_app.py
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app import create_app


def main():
    """Run the Flask application."""
    parser = argparse.ArgumentParser(description="ToF Gesture Classification Flask App")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Creating Flask application...")
        app = create_app()
        
        logger.info(f"Starting server on http://{args.host}:{args.port}")
        
        if not args.no_browser and args.host == "0.0.0.0":
            logger.info("Note: Browser will not auto-open when binding to 0.0.0.0")
            logger.info("       Use --host localhost to enable auto-browser")
        elif not args.no_browser:
            try:
                import webbrowser
                webbrowser.open(f"http://{args.host}:{args.port}")
            except Exception as e:
                logger.warning(f"Could not open browser: {e}")
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            use_reloader=args.debug
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
