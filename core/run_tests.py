#!/usr/bin/env python3
"""
Test Runner - Tüm testleri çalıştır
"""

import sys
import os
import subprocess

def run_tests():
    """Tüm testleri çalıştır"""
    print("🧪 Advanced AI Chatbot - Test Suite")
    print("=" * 50)
    
    test_commands = [
        {
            'name': 'Veri Kaynakları Testi',
            'command': 'python ../tests/test_advanced.py data',
            'description': 'Veri dosyalarının mevcut olup olmadığını kontrol eder'
        },
        {
            'name': 'Model Boyutları Testi', 
            'command': 'python ../tests/test_advanced.py models',
            'description': 'Farklı model boyutlarını test eder'
        },
        {
            'name': 'Konuşma Akışı Testi',
            'command': 'python ../tests/test_advanced.py conversation',
            'description': 'Chatbot konuşma akışını test eder'
        }
    ]
    
    results = []
    
    for test in test_commands:
        print(f"\n🔍 {test['name']}")
        print(f"📝 {test['description']}")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                test['command'].split(),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print("✅ Test Başarılı")
                results.append(('✅', test['name']))
            else:
                print("❌ Test Başarısız")
                print(f"Hata: {result.stderr}")
                results.append(('❌', test['name']))
                
        except subprocess.TimeoutExpired:
            print("⏰ Test Zaman Aşımı")
            results.append(('⏰', test['name']))
        except Exception as e:
            print(f"💥 Test Hatası: {e}")
            results.append(('💥', test['name']))
    
    # Sonuçları özetle
    print("\n" + "=" * 50)
    print("📊 Test Sonuçları:")
    
    for status, name in results:
        print(f"   {status} {name}")
    
    success_count = sum(1 for status, _ in results if status == '✅')
    total_count = len(results)
    
    print(f"\n🎯 Başarı Oranı: {success_count}/{total_count} (%{(success_count/total_count)*100:.1f})")
    
    return success_count == total_count

def run_interactive_test():
    """İnteraktif test modunu başlat"""
    print("🎮 İnteraktif Test Modu Başlatılıyor...")
    try:
        subprocess.run(['python', '../tests/test_advanced.py', 'interactive'])
    except KeyboardInterrupt:
        print("\n👋 Test sonlandırıldı.")
    except Exception as e:
        print(f"❌ İnteraktif test hatası: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_interactive_test()
    else:
        success = run_tests()
        sys.exit(0 if success else 1)