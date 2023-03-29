import os
from datetime import datetime
# Use the package we installed
from slack_bolt.async_app import AsyncApp
from dotenv import load_dotenv
import logging
from question_answering_system.inference import answer_query_with_context

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

# Initializes your app with your bot token and signing secret
app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

# Add functionality here
@app.event("app_home_opened")
async def update_home_tab(client, event, logger):

    try:
        update_index_button =  {
            "type": "button",
            "action_id": "update_index",
            "text": {
                "type": "plain_text",
                "text": "Update Index"
            }
        }
        update_index_action =   {
            "type": "actions",
            "elements": [update_index_button]
            }
        view = {
            "type": "home",
            "callback_id": "home_view",

            # body of the view
            "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Welcome to DeepDoc Home :book:"
                }
            }
            ]
        }

        if event["user"] == "U04U83NENR5":
            view["blocks"] = view["blocks"] + [update_index_action]
        
        # views.publish is the method that your app uses to push a view to the Home tab
        await client.views_publish(
            # the user that opened your app's app home
            user_id=event["user"],
            # the view object that appears in the app home
            view=view
        )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.action("update_index")
async def handle_update_index(ack, body, client, logger):

    await ack()
    
    update_index_button =  {
        "type": "button",
        "action_id": "update_index",
        "text": {
            "type": "plain_text",
            "text": "Update Index"
        }
    }
    update_index_action =   {
        "type": "actions",
        "elements": [update_index_button]
    }
    view = {
        "type": "home",
        "callback_id": "home_view",

        # body of the view
        "blocks": [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Welcome to DeepDoc Home :book:"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Index updated at {datetime.fromtimestamp(int(float(body['actions'][0]['action_ts'])))}"
            }
        }
        ]
    }

    if body["user"]["id"] == "U04U83NENR5":
        view["blocks"] = view["blocks"] + [update_index_action]
    
    try:
        await client.views_update(
            # the user that opened your app's app home
            view_id=body["view"]["id"],
            # the view object that appears in the app home
            view=view
            )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


@app.message()
async def handle_message(message, say):
    query = message["text"]
    answer = answer_query_with_context(query=query)
    await say(answer)
    

# Start your app
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))
