# Phishing Websites dataset — feature guide in plain language

The uploaded ARFF file contains 30 input features and one target column (`Result`). The descriptions below are short explanations based on the accompanying *Phishing Websites Features* guide supplied with the assignment materials. They are intended to help understand the column names; they are not a replacement for the original feature definitions.

| No. | ARFF column | Plain-language meaning |
|---:|---|---|
| 1 | `having_IP_Address` | Whether the address uses an IP address instead of an ordinary domain name. |
| 2 | `URL_Length` | Whether the URL length is considered normal, suspicious or long. |
| 3 | `Shortining_Service` | Whether a URL-shortening service is used. |
| 4 | `having_At_Symbol` | Whether the URL contains an `@` symbol. |
| 5 | `double_slash_redirecting` | Whether an unusual double-slash redirect pattern appears in the URL. |
| 6 | `Prefix_Suffix` | Whether a hyphen is used in the domain name in a potentially misleading way. |
| 7 | `having_Sub_Domain` | Whether the domain has subdomains, including multiple subdomains. |
| 8 | `SSLfinal_State` | Whether the SSL/HTTPS certificate state appears trustworthy. |
| 9 | `Domain_registeration_length` | Whether the domain registration period is considered short or long. |
| 10 | `Favicon` | Whether the favicon is loaded from the expected website/domain. |
| 11 | `port` | Whether the website uses a standard or non-standard network port. |
| 12 | `HTTPS_token` | Whether an HTTPS token is used misleadingly within the domain portion of the URL. |
| 13 | `Request_URL` | Whether page objects such as images or media are requested from another domain. |
| 14 | `URL_of_Anchor` | Whether links in page anchors point to appropriate destinations. |
| 15 | `Links_in_tags` | Whether links in Meta, Script and Link tags are consistent with the website domain. |
| 16 | `SFH` | Server Form Handler behaviour: where submitted form information is sent. |
| 17 | `Submitting_to_email` | Whether a form submits information using an email-related mechanism. |
| 18 | `Abnormal_URL` | Whether the URL/identity relationship appears abnormal. |
| 19 | `Redirect` | Website forwarding/redirect behaviour. |
| 20 | `on_mouseover` | Whether mouse-over behaviour changes or disguises the status-bar link. |
| 21 | `RightClick` | Whether the page disables the browser's right-click function. |
| 22 | `popUpWidnow` | Whether pop-up windows are used in a potentially suspicious way. |
| 23 | `Iframe` | Whether an iframe is used in a potentially suspicious way. |
| 24 | `age_of_domain` | Whether the domain is considered old enough to appear established. |
| 25 | `DNSRecord` | Whether the domain has a recognised DNS record. |
| 26 | `web_traffic` | Whether the website has a level of traffic/popularity consistent with an established site. |
| 27 | `Page_Rank` | Whether the page has a meaningful PageRank value. |
| 28 | `Google_Index` | Whether the website is indexed by Google. |
| 29 | `Links_pointing_to_page` | Number/category of links pointing to the webpage. |
| 30 | `Statistical_report` | Whether the website appears in statistical reports associated with phishing activity. |

## Understanding the coded values

The ARFF file uses integer codes such as `-1`, `0` and `+1`. These codes are assigned according to each feature's own rule. **Do not assume that `-1`, `0` and `+1` have the same plain-language meaning for every feature.** The target column is separate: in this dataset version, `Result = -1` represents Phishing and `Result = +1` represents Legitimate.
