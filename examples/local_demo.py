from prefill_jev.script import system_one

dic = {
    "state": "Customer wants to return a damaged product.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Choose the department that should handle this request.",
            "criteria": {
                "returns": "Exchanges, refunds, wrong or damaged items",
                "shipping": "Delivery status, delays, lost packages",
                "billing": "Charges, invoices, payment problems"
            }
        },

        "escalate": {
            "type": "noul",
            "instructions": "Does this message request a refund?"
        },

        "frustration": {
            "type": "score",
            "instructions": "Rate the customer's level of frustration.",
            "criteria": ["Calm", "Frustrated", "Very angry"]
        }
    }
}

result = system_one(dic["state"], dic["questions"])
print(result)

