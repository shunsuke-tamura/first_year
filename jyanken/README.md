# じゃんけん絶対勝つやつ
## 背景
とある理由で成果物が必要になったため、高校のときにつくった簡易筆跡鑑定を参考に作った

## 仕様
- じゃんけんを3回行う
- 毎回のじゃんけんの前に1つ質問を投げる
- 質問はyes/noで答えられるもので、yesならチョキ、noならパーで回答する
- 3回目のじゃんけんで必ず勝つようになっている

## ロジック
簡単に言うと「究極の後出し」  
- AIにグー、チョキ、パーの他に、チョキになる数フレーム前の手、パーになる前の数フレーム前の手を学習させる
- じゃんけん前にまったく関係のない質問をすることであたかも質問な回答から次の手を予測しているように装う
- こちらからじゃんけんの音声を流すことでタイミングをなんとなく覚えてもらう
- 3回目だけコンマ数秒「ポン」のタイミングを遅くすることで人間側に微妙に早出しさせる
- AIでそのコンマ数秒の差によって生まれる「何かになる前の手」から何を出そうとしているのかを判別
- 究極の後出しによって3回目だけ必ず勝つようになっている
- 1回戦、2回戦でAIが出す手は完全ランダム