# TestFlight Upload

CAIX Chat is configured for automatic signing on Apple team `34UZ7Y2KSW` with bundle ID `com.redhillsmediafl.CaixChat`.

## One-Time App Store Connect Setup

The upload script runs the Fastlane lane `ios ensure_app_record` before archiving. That lane uses the App Store Connect API key to verify/register the Bundle ID, then verifies the App Store Connect app record:

- Name: `CAIX Chat`
- Bundle ID: `com.redhillsmediafl.CaixChat`
- SKU: `caix-chat-ios-0001`
- Primary language: English (U.S.)

If the App Store Connect app record is missing, Fastlane can create it through `produce`, but that path requires an Apple ID web session rather than only the API key. Set `FASTLANE_USER`, `APPLE_ID`, or `APP_STORE_CONNECT_USERNAME` before running the lane/script; Fastlane will use an existing session or prompt for Apple ID authentication.

You can run the setup step by itself from `Clients/CaixChat`:

```bash
ASC_KEY_PATH=/path/to/AuthKey_KEYID.p8 \
ASC_KEY_ID=KEYID \
ASC_ISSUER_ID=issuer-uuid \
FASTLANE_USER=apple-id@example.com \
INTERNAL_TESTER_EMAIL=person@example.com \
fastlane ios ensure_app_record
```

The full upload script does this automatically. Xcode's command-line upload refuses the archive until the app record exists, so the script also verifies the record with `altool --list-apps` after Fastlane runs.

## Upload

From the repo root:

```bash
Clients/CaixChat/Scripts/upload-testflight.sh
```

The script requires an App Store Connect API key and the internal TestFlight tester email:

```bash
ASC_KEY_PATH=/path/to/AuthKey_KEYID.p8 \
ASC_KEY_ID=KEYID \
ASC_ISSUER_ID=issuer-uuid \
INTERNAL_TESTER_EMAIL=person@example.com \
Clients/CaixChat/Scripts/upload-testflight.sh
```

The upload path is:

1. Ensure the Bundle ID and App Store Connect app record exist via Fastlane.
2. Verify the App Store Connect app record with `altool --list-apps`.
3. Archive `CaixChat.xcodeproj` with automatic signing.
4. Export and upload using `ExportOptions.plist` with `method=app-store-connect` and `destination=upload`.
5. Set TestFlight export compliance, create/use the internal TestFlight group, and assign the uploaded build to `INTERNAL_TESTER_EMAIL`.

Override the internal tester if needed:

```bash
INTERNAL_TESTER_EMAIL=person@example.com \
ASC_KEY_PATH=/path/to/AuthKey_KEYID.p8 \
ASC_KEY_ID=KEYID \
ASC_ISSUER_ID=issuer-uuid \
Clients/CaixChat/Scripts/upload-testflight.sh
```
