# 🌍 Multilingual Dashboard - English & Korean Support

## ✨ What's New

Your SPECTRA AI dashboard now supports **three languages**:
- 🇺🇸 **English**
- 🇷🇺 **Русский (Russian)** 
- 🇰🇷 **한국어 (Korean)**

## 🔄 How to Switch Languages

1. **Open the dashboard** at: `http://localhost:5000`
2. **Look for the language selector** in the top-right corner (next to the connection status)
3. **Select your preferred language**:
   - 🇺🇸 English
   - 🇷🇺 Русский 
   - 🇰🇷 한국어
4. The interface will **automatically reload** with the selected language

## 💾 Language Preference

Your language choice is **saved automatically** to your browser's local storage. Next time you open the dashboard, it will remember your preferred language!

## 📱 Supported Features in All Languages

✅ Monitoring interface (real-time analysis)
✅ Dashboard (statistics & charts)
✅ Session history
✅ Face recognition system
✅ Health analysis (DOC.FAI.ME)
✅ Settings & configuration
✅ All buttons, labels, and messages

## 🇰🇷 Korean Language Features

- Complete UI translation to Korean
- Full support for Korean character rendering
- All metrics and labels in Korean
- Voice commands and descriptions in Korean

## 🇺🇸 English Translation

- Professional English terminology
- Clear, concise descriptions
- Consistent with scientific health monitoring standards

## 📋 Backup Files

Your original Russian dashboard has been backed up as:
- `app_dashboard_ru.html` - Original Russian version (2,828 lines)
- `app_dashboard_multilang.html` - Alternative multilingual version

You can restore the original by:
```bash
cp app_dashboard_ru.html app_dashboard.html
```

## 🛠️ Technical Details

The multilingual system works by:
1. **Language Selector** - Located in the header next to connection status
2. **Translation Engine** - JSON-based translation mapping for 100+ UI strings
3. **Local Storage** - Saves your language preference
4. **Page Reload** - Applies the selected language

### Supported Languages by Code:
- `en` = English
- `ru` = Русский
- `ko` = 한국어

## 🚀 How to Run

```bash
# Start the system
./app-start.sh

# Then open in your browser:
# http://localhost:5000
```

## 🐛 Troubleshooting

### Language selector not showing?
- Clear browser cache: `Cmd+Shift+Delete`
- Refresh the page: `Cmd+R`

### Characters not displaying correctly?
- Make sure your browser supports Unicode (all modern browsers do)
- Check that your OS supports Korean fonts (macOS, Windows, Linux all have support)

### Language didn't change?
- Check that JavaScript is enabled
- Try switching languages again
- Clear local storage and reset

## 📞 Support

If you need to revert to the Russian-only version:
```bash
cp app_dashboard_ru.html app_dashboard.html
./app-start.sh
```

---

**Enjoy SPECTRA AI in your preferred language!** 🎉
