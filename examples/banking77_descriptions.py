"""Example: 77 generic per-class descriptions for Banking77 — the "+quality" label-text path.

Each is a plain one-line rephrasing of the intent name (anyone can write these; nothing is taken
from the dataset). Pass them as `label_texts` (keys = the class names you pass as `labels`).
Measured on the full Banking77 test (3,080), same 3-model ensemble: names 0.7555 -> descriptions
0.7737 (+0.018). Reproducible in this repo. (Jev, an independent reproduction, scores ~0.801 zero-shot;
these descriptions narrow the gap but do NOT overtake it — Quorum beats Jev at the 24-shot budget, not
zero-shot. See docs/phases/1.1/ and the README.)

Run:  python examples/banking77_descriptions.py
"""
from quorum import ZeroShotEnsemble

# name -> one-line generic description
BANKING77_DESCRIPTIONS = {
    'activate_my_card': 'activating a card they just received',
    'age_limit': 'the minimum age required to open an account',
    'apple_pay_or_google_pay': 'using Apple Pay or Google Pay',
    'atm_support': 'whether they can use an ATM',
    'automatic_top_up': 'setting up automatic top-ups',
    'balance_not_updated_after_bank_transfer': 'a balance not updating after a bank transfer',
    'balance_not_updated_after_cheque_or_cash_deposit': 'a balance not updating after a cheque or cash deposit',
    'beneficiary_not_allowed': 'a beneficiary or payee not being allowed',
    'cancel_transfer': 'cancelling a transfer',
    'card_about_to_expire': 'a card that is about to expire',
    'card_acceptance': 'where the card is accepted',
    'card_arrival': 'when a new card will arrive',
    'card_delivery_estimate': 'an estimated delivery time for a card',
    'card_linking': 'linking a card to the account',
    'card_not_working': 'a card that is not working',
    'card_payment_fee_charged': 'a fee charged on a card payment',
    'card_payment_not_recognised': 'a card payment they do not recognise',
    'card_payment_wrong_exchange_rate': 'a wrong exchange rate on a card payment',
    'card_swallowed': 'a card swallowed by an ATM',
    'cash_withdrawal_charge': 'a charge on a cash withdrawal',
    'cash_withdrawal_not_recognised': 'a cash withdrawal they do not recognise',
    'change_pin': 'changing the card PIN',
    'compromised_card': 'a compromised or hacked card',
    'contactless_not_working': 'contactless payments not working',
    'country_support': 'which countries are supported',
    'declined_card_payment': 'a declined card payment',
    'declined_cash_withdrawal': 'a declined cash withdrawal',
    'declined_transfer': 'a declined transfer',
    'direct_debit_payment_not_recognised': 'a direct debit payment they do not recognise',
    'disposable_card_limits': 'limits on disposable virtual cards',
    'edit_personal_details': 'editing personal details',
    'exchange_charge': 'a charge for exchanging currency',
    'exchange_rate': 'the exchange rate that is used',
    'exchange_via_app': 'exchanging currency in the app',
    'extra_charge_on_statement': 'an unexpected extra charge on the statement',
    'failed_transfer': 'a transfer that failed',
    'fiat_currency_support': 'which traditional (fiat) currencies are supported',
    'get_disposable_virtual_card': 'getting a disposable virtual card',
    'get_physical_card': 'getting a physical card',
    'getting_spare_card': 'getting a spare card',
    'getting_virtual_card': 'getting a virtual card',
    'lost_or_stolen_card': 'a lost or stolen card',
    'lost_or_stolen_phone': 'a lost or stolen phone',
    'order_physical_card': 'ordering a physical card',
    'passcode_forgotten': 'a forgotten passcode or password',
    'pending_card_payment': 'a card payment that is still pending',
    'pending_cash_withdrawal': 'a cash withdrawal that is still pending',
    'pending_top_up': 'a top-up that is still pending',
    'pending_transfer': 'a transfer that is still pending',
    'pin_blocked': 'a blocked PIN',
    'receiving_money': 'receiving money from someone',
    'Refund_not_showing_up': 'a refund that is not showing up',
    'request_refund': 'requesting a refund',
    'reverted_card_payment?': 'a card payment that was reverted or reversed',
    'supported_cards_and_currencies': 'which cards and currencies are supported',
    'terminate_account': 'closing or terminating the account',
    'top_up_by_bank_transfer_charge': 'a charge for topping up by bank transfer',
    'top_up_by_card_charge': 'a charge for topping up by card',
    'top_up_by_cash_or_cheque': 'topping up by cash or cheque',
    'top_up_failed': 'a top-up that failed',
    'top_up_limits': 'limits on how much can be topped up',
    'top_up_reverted': 'a top-up that was reverted or reversed',
    'topping_up_by_card': 'topping up the account by card',
    'transaction_charged_twice': 'a transaction that was charged twice',
    'transfer_fee_charged': 'a fee charged on a transfer',
    'transfer_into_account': 'transferring money into the account',
    'transfer_not_received_by_recipient': 'a transfer not received by the recipient',
    'transfer_timing': 'how long a transfer takes to arrive',
    'unable_to_verify_identity': 'being unable to verify their identity',
    'verify_my_identity': 'verifying their identity',
    'verify_source_of_funds': 'verifying the source of their funds',
    'verify_top_up': 'verifying a top-up',
    'virtual_card_not_working': 'a virtual card not working',
    'visa_or_mastercard': 'choosing between Visa and Mastercard',
    'why_verify_identity': 'why identity verification is required',
    'wrong_amount_of_cash_received': 'receiving the wrong amount of cash from an ATM',
    'wrong_exchange_rate_for_cash_withdrawal': 'a wrong exchange rate on a cash withdrawal',
}

if __name__ == "__main__":
    labels = list(BANKING77_DESCRIPTIONS)
    descriptions = [BANKING77_DESCRIPTIONS[name] for name in labels]
    clf = ZeroShotEnsemble()
    text = "my card still hasn't come"
    proba = clf.predict_proba(text, labels, label_texts=descriptions)
    ranked = sorted(zip(labels, proba), key=lambda kv: kv[1], reverse=True)
    print(f"text: {text!r}")
    for name, p in ranked[:3]:
        print(f"  {p:6.3f}  {name}")
